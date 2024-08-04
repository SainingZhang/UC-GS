#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
import pickle
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_ucloss, uc_ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    uc_file_path = '/DATA_EDS2/yebj/uc_gs/Scaffold-GS/uc/uc18_mm3.pkl'

    with open(uc_file_path, 'rb') as file:
        weight_dict = pickle.load(file)
    
    weights_dict_cuda = {}
    # check cuda
    if torch.cuda.is_available():

        for name, uc_weight in weight_dict.items():
            weight_cuda = uc_weight.to('cuda')
            weights_dict_cuda[name] = weight_cuda
            
    else:
        print("No cuda!")


    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        weight = weights_dict_cuda[viewpoint_cam.image_name]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_ucloss(image, gt_image, weight)
        #Ll1 = l1_loss(image, gt_image)

        ssim_loss = uc_ssim(image, gt_image, weight)
        #ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        #loss = (1.0 - opt.lambda_dssim) * uc_Ll1 + opt.lambda_dssim * uc_ssim_loss + 0.01*scaling_reg


        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[900000, 902000, 904000, 906000, 908000, 910000, 912000, 914000, 916000, 918000, 920000, 922000, 924000, 926000, 928000, 930000, 932000, 934000, 936000, 938000, 940000, 942000, 944000, 946000, 948000, 950000, 952000, 954000, 956000, 958000, 960000, 962000, 964000, 966000, 968000, 970000, 972000, 974000, 976000, 978000, 980000, 982000, 984000, 986000, 988000, 990000, 992000, 994000, 996000, 998000, 1000000, 1002000, 1004000, 1006000, 1008000, 1010000, 1012000, 1014000, 1016000, 1018000, 1020000, 1022000, 1024000, 1026000, 1028000, 1030000, 1032000, 1034000, 1036000, 1038000, 1040000, 1042000, 1044000, 1046000, 1048000, 1050000, 1052000, 1054000, 1056000, 1058000, 1060000, 1062000, 1064000, 1066000, 1068000, 1070000, 1072000, 1074000, 1076000, 1078000, 1080000, 1082000, 1084000, 1086000, 1088000, 1090000, 1092000, 1094000, 1096000, 1098000, 1100000, 1102000, 1104000, 1106000, 1108000, 1110000, 1112000, 1114000, 1116000, 1118000, 1120000, 1122000, 1124000, 1126000, 1128000, 1130000, 1132000, 1134000, 1136000, 1138000, 1140000, 1142000, 1144000, 1146000, 1148000, 1150000, 1152000, 1154000, 1156000, 1158000, 1160000, 1162000, 1164000, 1166000, 1168000, 1170000, 1172000, 1174000, 1176000, 1178000, 1180000, 1182000, 1184000, 1186000, 1188000, 1190000, 1192000, 1194000, 1196000, 1198000, 1200000, 1202000, 1204000, 1206000, 1208000, 1210000, 1212000, 1214000, 1216000, 1218000, 1220000, 1222000, 1224000, 1226000, 1228000, 1230000, 1232000, 1234000, 1236000, 1238000, 1240000, 1242000, 1244000, 1246000, 1248000, 1250000, 1252000, 1254000, 1256000, 1258000, 1260000, 1262000, 1264000, 1266000, 1268000, 1270000, 1272000, 1274000, 1276000, 1278000, 1280000, 1282000, 1284000, 1286000, 1288000, 1290000, 1292000, 1294000, 1296000, 1298000, 1300000, 1302000, 1304000, 1306000, 1308000, 1310000, 1312000, 1314000, 1316000, 1318000, 1320000, 1322000, 1324000, 1326000, 1328000, 1330000, 1332000, 1334000, 1336000, 1338000, 1340000, 1342000, 1344000, 1346000, 1348000, 1350000, 1352000, 1354000, 1356000, 1358000, 1360000, 1362000, 1364000, 1366000, 1368000, 1370000, 1372000, 1374000, 1376000, 1378000, 1380000, 1382000, 1384000, 1386000, 1388000, 1390000, 1392000, 1394000, 1396000, 1398000, 1400000, 1402000, 1404000, 1406000, 1408000, 1410000, 1412000, 1414000, 1416000, 1418000, 1420000, 1422000, 1424000, 1426000, 1428000, 1430000, 1432000, 1434000, 1436000, 1438000, 1440000, 1442000, 1444000, 1446000, 1448000, 1450000, 1452000, 1454000, 1456000, 1458000, 1460000, 1462000, 1464000, 1466000, 1468000, 1470000, 1472000, 1474000, 1476000, 1478000, 1480000, 1482000, 1484000, 1486000, 1488000, 1490000, 1492000, 1494000, 1496000, 1498000, 1500000, 1502000, 1504000, 1506000, 1508000, 1510000, 1512000, 1514000, 1516000, 1518000, 1520000, 1522000, 1524000, 1526000, 1528000, 1530000, 1532000, 1534000, 1536000, 1538000, 1540000, 1542000, 1544000, 1546000, 1548000, 1550000, 1552000, 1554000, 1556000, 1558000, 1560000, 1562000, 1564000, 1566000, 1568000, 1570000, 1572000, 1574000, 1576000, 1578000, 1580000, 1582000, 1584000, 1586000, 1588000, 1590000, 1592000, 1594000, 1596000, 1598000, 1600000, 1602000, 1604000, 1606000, 1608000, 1610000, 1612000, 1614000, 1616000, 1618000, 1620000, 1622000, 1624000, 1626000, 1628000, 1630000, 1632000, 1634000, 1636000, 1638000, 1640000, 1642000, 1644000, 1646000, 1648000, 1650000, 1652000, 1654000, 1656000, 1658000, 1660000, 1662000, 1664000, 1666000, 1668000, 1670000, 1672000, 1674000, 1676000, 1678000, 1680000, 1682000, 1684000, 1686000, 1688000, 1690000, 1692000, 1694000, 1696000, 1698000, 1700000, 1702000, 1704000, 1706000, 1708000, 1710000, 1712000, 1714000, 1716000, 1718000, 1720000, 1722000, 1724000, 1726000, 1728000, 1730000, 1732000, 1734000, 1736000, 1738000, 1740000, 1742000, 1744000, 1746000, 1748000, 1750000, 1752000, 1754000, 1756000, 1758000, 1760000, 1762000, 1764000, 1766000, 1768000, 1770000, 1772000, 1774000, 1776000, 1778000, 1780000, 1782000, 1784000, 1786000, 1788000, 1790000, 1792000, 1794000, 1796000, 1798000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[900000, 902000, 904000, 906000, 908000, 910000, 912000, 914000, 916000, 918000, 920000, 922000, 924000, 926000, 928000, 930000, 932000, 934000, 936000, 938000, 940000, 942000, 944000, 946000, 948000, 950000, 952000, 954000, 956000, 958000, 960000, 962000, 964000, 966000, 968000, 970000, 972000, 974000, 976000, 978000, 980000, 982000, 984000, 986000, 988000, 990000, 992000, 994000, 996000, 998000, 1000000, 1002000, 1004000, 1006000, 1008000, 1010000, 1012000, 1014000, 1016000, 1018000, 1020000, 1022000, 1024000, 1026000, 1028000, 1030000, 1032000, 1034000, 1036000, 1038000, 1040000, 1042000, 1044000, 1046000, 1048000, 1050000, 1052000, 1054000, 1056000, 1058000, 1060000, 1062000, 1064000, 1066000, 1068000, 1070000, 1072000, 1074000, 1076000, 1078000, 1080000, 1082000, 1084000, 1086000, 1088000, 1090000, 1092000, 1094000, 1096000, 1098000, 1100000, 1102000, 1104000, 1106000, 1108000, 1110000, 1112000, 1114000, 1116000, 1118000, 1120000, 1122000, 1124000, 1126000, 1128000, 1130000, 1132000, 1134000, 1136000, 1138000, 1140000, 1142000, 1144000, 1146000, 1148000, 1150000, 1152000, 1154000, 1156000, 1158000, 1160000, 1162000, 1164000, 1166000, 1168000, 1170000, 1172000, 1174000, 1176000, 1178000, 1180000, 1182000, 1184000, 1186000, 1188000, 1190000, 1192000, 1194000, 1196000, 1198000, 1200000, 1202000, 1204000, 1206000, 1208000, 1210000, 1212000, 1214000, 1216000, 1218000, 1220000, 1222000, 1224000, 1226000, 1228000, 1230000, 1232000, 1234000, 1236000, 1238000, 1240000, 1242000, 1244000, 1246000, 1248000, 1250000, 1252000, 1254000, 1256000, 1258000, 1260000, 1262000, 1264000, 1266000, 1268000, 1270000, 1272000, 1274000, 1276000, 1278000, 1280000, 1282000, 1284000, 1286000, 1288000, 1290000, 1292000, 1294000, 1296000, 1298000, 1300000, 1302000, 1304000, 1306000, 1308000, 1310000, 1312000, 1314000, 1316000, 1318000, 1320000, 1322000, 1324000, 1326000, 1328000, 1330000, 1332000, 1334000, 1336000, 1338000, 1340000, 1342000, 1344000, 1346000, 1348000, 1350000, 1352000, 1354000, 1356000, 1358000, 1360000, 1362000, 1364000, 1366000, 1368000, 1370000, 1372000, 1374000, 1376000, 1378000, 1380000, 1382000, 1384000, 1386000, 1388000, 1390000, 1392000, 1394000, 1396000, 1398000, 1400000, 1402000, 1404000, 1406000, 1408000, 1410000, 1412000, 1414000, 1416000, 1418000, 1420000, 1422000, 1424000, 1426000, 1428000, 1430000, 1432000, 1434000, 1436000, 1438000, 1440000, 1442000, 1444000, 1446000, 1448000, 1450000, 1452000, 1454000, 1456000, 1458000, 1460000, 1462000, 1464000, 1466000, 1468000, 1470000, 1472000, 1474000, 1476000, 1478000, 1480000, 1482000, 1484000, 1486000, 1488000, 1490000, 1492000, 1494000, 1496000, 1498000, 1500000, 1502000, 1504000, 1506000, 1508000, 1510000, 1512000, 1514000, 1516000, 1518000, 1520000, 1522000, 1524000, 1526000, 1528000, 1530000, 1532000, 1534000, 1536000, 1538000, 1540000, 1542000, 1544000, 1546000, 1548000, 1550000, 1552000, 1554000, 1556000, 1558000, 1560000, 1562000, 1564000, 1566000, 1568000, 1570000, 1572000, 1574000, 1576000, 1578000, 1580000, 1582000, 1584000, 1586000, 1588000, 1590000, 1592000, 1594000, 1596000, 1598000, 1600000, 1602000, 1604000, 1606000, 1608000, 1610000, 1612000, 1614000, 1616000, 1618000, 1620000, 1622000, 1624000, 1626000, 1628000, 1630000, 1632000, 1634000, 1636000, 1638000, 1640000, 1642000, 1644000, 1646000, 1648000, 1650000, 1652000, 1654000, 1656000, 1658000, 1660000, 1662000, 1664000, 1666000, 1668000, 1670000, 1672000, 1674000, 1676000, 1678000, 1680000, 1682000, 1684000, 1686000, 1688000, 1690000, 1692000, 1694000, 1696000, 1698000, 1700000, 1702000, 1704000, 1706000, 1708000, 1710000, 1712000, 1714000, 1716000, 1718000, 1720000, 1722000, 1724000, 1726000, 1728000, 1730000, 1732000, 1734000, 1736000, 1738000, 1740000, 1742000, 1744000, 1746000, 1748000, 1750000, 1752000, 1754000, 1756000, 1758000, 1760000, 1762000, 1764000, 1766000, 1768000, 1770000, 1772000, 1774000, 1776000, 1778000, 1780000, 1782000, 1784000, 1786000, 1788000, 1790000, 1792000, 1794000, 1796000, 1798000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,50_000,52000,54000,56000,58000,60_000,62000,64000,66000,68000,70_000,71000,72000,73000,74000,75000,76000,77000,78000,79000,80_000,81000,82000,83000,84000,85000,86000,87000,88000,89000,90_000,91000,92000,93000,94000,95000,96000,97000,98000,99000,
    100_000,101000,102000,103000,104000,105000,106000,107000,108000,109000,110_000,111000,112000,113000,114000,115000,116000,117000,118000,119000,120_000,121000,122000,123000,124000,125000,126000,127000,128000,129000,130_000,131000,132000,133000,134000,135000,136000,137000,138000,139000,
    140_000,141000,142000,143000,144000,145000,146000,147000,148000,149000,150_000,151000,152000,153000,154000,155000,156000,157000,158000,159000,160_000,162000,164000,166000,168000,170_000,172000,174000,176000,178000,180_000,182000,184000,186000,188000,
    190_000,192000,194000,196000,198000,200_000,202000,204000,206000,208000,210_000,212000,214000,216000,218000,220_000,222000,224000,226000,228000,230_000,232000,234000,236000,238000,
    240_000,242000,244000,246000,248000,250_000,252000,254000,256000,258000,260_000,262000,264000,266000,268000,270_000,272000,274000,276000,278000,280_000,282000,284000,286000,288000,290_000,292000,294000,296000,298000,300000, 302000, 304000, 306000, 308000, 310000, 312000, 314000, 316000, 318000, 320000, 322000, 324000, 326000, 328000, 330000, 332000, 334000, 336000, 338000, 340000, 342000, 344000, 346000, 348000, 350000, 352000, 354000, 356000, 358000, 360000, 362000, 364000, 366000, 368000, 370000, 372000, 374000, 376000, 378000, 380000, 382000, 384000, 386000, 388000, 390000, 392000, 394000, 396000, 398000, 400000, 402000, 404000, 406000, 408000, 410000, 412000, 414000, 416000, 418000, 420000, 422000, 424000, 426000, 428000, 430000, 432000, 434000, 436000, 438000, 440000, 442000, 444000, 446000, 448000, 450000, 452000, 454000, 456000, 458000, 460000, 462000, 464000, 466000, 468000, 470000, 472000, 474000, 476000, 478000, 480000, 482000, 484000, 486000, 488000, 490000, 492000, 494000, 496000, 498000, 500000, 502000, 504000, 506000, 508000, 510000, 512000, 514000, 516000, 518000, 520000, 522000, 524000, 526000, 528000, 530000, 532000, 534000, 536000, 538000, 540000, 542000, 544000, 546000, 548000, 550000, 552000, 554000, 556000, 558000, 560000, 562000, 564000, 566000, 568000, 570000, 572000, 574000, 576000, 578000, 580000, 582000, 584000, 586000, 588000, 590000, 592000, 594000, 596000, 598000,600000, 602000, 604000, 606000, 608000, 610000, 612000, 614000, 616000, 618000, 620000, 622000, 624000, 626000, 628000, 630000, 632000, 634000, 636000, 638000, 640000, 642000, 644000, 646000, 648000, 650000, 652000, 654000, 656000, 658000, 660000, 662000, 664000, 666000, 668000, 670000, 672000, 674000, 676000, 678000, 680000, 682000, 684000, 686000, 688000, 690000, 692000, 694000, 696000, 698000, 700000, 702000, 704000, 706000, 708000, 710000, 712000, 714000, 716000, 718000, 720000, 722000, 724000, 726000, 728000, 730000, 732000, 734000, 736000, 738000, 740000, 742000, 744000, 746000, 748000, 750000, 752000, 754000, 756000, 758000, 760000, 762000, 764000, 766000, 768000, 770000, 772000, 774000, 776000, 778000, 780000, 782000, 784000, 786000, 788000, 790000, 792000, 794000, 796000, 798000, 800000, 802000, 804000, 806000, 808000, 810000, 812000, 814000, 816000, 818000, 820000, 822000, 824000, 826000, 828000, 830000, 832000, 834000, 836000, 838000, 840000, 842000, 844000, 846000, 848000, 850000, 852000, 854000, 856000, 858000, 860000, 862000, 864000, 866000, 868000, 870000, 872000, 874000, 876000, 878000, 880000, 882000, 884000, 886000, 888000, 890000, 892000, 894000, 896000, 898000,900000, 902000, 904000, 906000, 908000, 910000, 912000, 914000, 916000, 918000, 920000, 922000, 924000, 926000, 928000, 930000, 932000, 934000, 936000, 938000, 940000, 942000, 944000, 946000, 948000, 950000, 952000, 954000, 956000, 958000, 960000, 962000, 964000, 966000, 968000, 970000, 972000, 974000, 976000, 978000, 980000, 982000, 984000, 986000, 988000, 990000, 992000, 994000, 996000, 998000, 1000000, 1002000, 1004000, 1006000, 1008000, 1010000, 1012000, 1014000, 1016000, 1018000, 1020000, 1022000, 1024000, 1026000, 1028000, 1030000, 1032000, 1034000, 1036000, 1038000, 1040000, 1042000, 1044000, 1046000, 1048000, 1050000, 1052000, 1054000, 1056000, 1058000, 1060000, 1062000, 1064000, 1066000, 1068000, 1070000, 1072000, 1074000, 1076000, 1078000, 1080000, 1082000, 1084000, 1086000, 1088000, 1090000, 1092000, 1094000, 1096000, 1098000, 1100000, 1102000, 1104000, 1106000, 1108000, 1110000, 1112000, 1114000, 1116000, 1118000, 1120000, 1122000, 1124000, 1126000, 1128000, 1130000, 1132000, 1134000, 1136000, 1138000, 1140000, 1142000, 1144000, 1146000, 1148000, 1150000, 1152000, 1154000, 1156000, 1158000, 1160000, 1162000, 1164000, 1166000, 1168000, 1170000, 1172000, 1174000, 1176000, 1178000, 1180000, 1182000, 1184000, 1186000, 1188000, 1190000, 1192000, 1194000, 1196000, 1198000, 1200000, 1202000, 1204000, 1206000, 1208000, 1210000, 1212000, 1214000, 1216000, 1218000, 1220000, 1222000, 1224000, 1226000, 1228000, 1230000, 1232000, 1234000, 1236000, 1238000, 1240000, 1242000, 1244000, 1246000, 1248000, 1250000, 1252000, 1254000, 1256000, 1258000, 1260000, 1262000, 1264000, 1266000, 1268000, 1270000, 1272000, 1274000, 1276000, 1278000, 1280000, 1282000, 1284000, 1286000, 1288000, 1290000, 1292000, 1294000, 1296000, 1298000, 1300000, 1302000, 1304000, 1306000, 1308000, 1310000, 1312000, 1314000, 1316000, 1318000, 1320000, 1322000, 1324000, 1326000, 1328000, 1330000, 1332000, 1334000, 1336000, 1338000, 1340000, 1342000, 1344000, 1346000, 1348000, 1350000, 1352000, 1354000, 1356000, 1358000, 1360000, 1362000, 1364000, 1366000, 1368000, 1370000, 1372000, 1374000, 1376000, 1378000, 1380000, 1382000, 1384000, 1386000, 1388000, 1390000, 1392000, 1394000, 1396000, 1398000, 1400000, 1402000, 1404000, 1406000, 1408000, 1410000, 1412000, 1414000, 1416000, 1418000, 1420000, 1422000, 1424000, 1426000, 1428000, 1430000, 1432000, 1434000, 1436000, 1438000, 1440000, 1442000, 1444000, 1446000, 1448000, 1450000, 1452000, 1454000, 1456000, 1458000, 1460000, 1462000, 1464000, 1466000, 1468000, 1470000, 1472000, 1474000, 1476000, 1478000, 1480000, 1482000, 1484000, 1486000, 1488000, 1490000, 1492000, 1494000, 1496000, 1498000, 1500000, 1502000, 1504000, 1506000, 1508000, 1510000, 1512000, 1514000, 1516000, 1518000, 1520000, 1522000, 1524000, 1526000, 1528000, 1530000, 1532000, 1534000, 1536000, 1538000, 1540000, 1542000, 1544000, 1546000, 1548000, 1550000, 1552000, 1554000, 1556000, 1558000, 1560000, 1562000, 1564000, 1566000, 1568000, 1570000, 1572000, 1574000, 1576000, 1578000, 1580000, 1582000, 1584000, 1586000, 1588000, 1590000, 1592000, 1594000, 1596000, 1598000, 1600000, 1602000, 1604000, 1606000, 1608000, 1610000, 1612000, 1614000, 1616000, 1618000, 1620000, 1622000, 1624000, 1626000, 1628000, 1630000, 1632000, 1634000, 1636000, 1638000, 1640000, 1642000, 1644000, 1646000, 1648000, 1650000, 1652000, 1654000, 1656000, 1658000, 1660000, 1662000, 1664000, 1666000, 1668000, 1670000, 1672000, 1674000, 1676000, 1678000, 1680000, 1682000, 1684000, 1686000, 1688000, 1690000, 1692000, 1694000, 1696000, 1698000, 1700000, 1702000, 1704000, 1706000, 1708000, 1710000, 1712000, 1714000, 1716000, 1718000, 1720000, 1722000, 1724000, 1726000, 1728000, 1730000, 1732000, 1734000, 1736000, 1738000, 1740000, 1742000, 1744000, 1746000, 1748000, 1750000, 1752000, 1754000, 1756000, 1758000, 1760000, 1762000, 1764000, 1766000, 1768000, 1770000, 1772000, 1774000, 1776000, 1778000, 1780000, 1782000, 1784000, 1786000, 1788000, 1790000, 1792000, 1794000, 1796000, 1798000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,50_000,52000,54000,56000,58000,60_000,62000,64000,66000,68000,70_000,71000,72000,73000,74000,75000,76000,77000,78000,79000,80_000,81000,82000,83000,84000,85000,86000,87000,88000,89000,90_000,91000,92000,93000,94000,95000,96000,97000,98000,99000,
    100_000,101000,102000,103000,104000,105000,106000,107000,108000,109000,110_000,111000,112000,113000,114000,115000,116000,117000,118000,119000,120_000,121000,122000,123000,124000,125000,126000,127000,128000,129000,130_000,131000,132000,133000,134000,135000,136000,137000,138000,139000,
    140_000,141000,142000,143000,144000,145000,146000,147000,148000,149000,150_000,151000,152000,153000,154000,155000,156000,157000,158000,159000,160_000,162000,164000,166000,168000,170_000,172000,174000,176000,178000,180_000,182000,184000,186000,188000,
    190_000,192000,194000,196000,198000,200_000,202000,204000,206000,208000,210_000,212000,214000,216000,218000,220_000,222000,224000,226000,228000,230_000,232000,234000,236000,238000,
    240_000,242000,244000,246000,248000,250_000,252000,254000,256000,258000,260_000,262000,264000,266000,268000,270_000,272000,274000,276000,278000,280_000,282000,284000,286000,288000,290_000,292000,294000,296000,298000,300000, 302000, 304000, 306000, 308000, 310000, 312000, 314000, 316000, 318000, 320000, 322000, 324000, 326000, 328000, 330000, 332000, 334000, 336000, 338000, 340000, 342000, 344000, 346000, 348000, 350000, 352000, 354000, 356000, 358000, 360000, 362000, 364000, 366000, 368000, 370000, 372000, 374000, 376000, 378000, 380000, 382000, 384000, 386000, 388000, 390000, 392000, 394000, 396000, 398000, 400000, 402000, 404000, 406000, 408000, 410000, 412000, 414000, 416000, 418000, 420000, 422000, 424000, 426000, 428000, 430000, 432000, 434000, 436000, 438000, 440000, 442000, 444000, 446000, 448000, 450000, 452000, 454000, 456000, 458000, 460000, 462000, 464000, 466000, 468000, 470000, 472000, 474000, 476000, 478000, 480000, 482000, 484000, 486000, 488000, 490000, 492000, 494000, 496000, 498000, 500000, 502000, 504000, 506000, 508000, 510000, 512000, 514000, 516000, 518000, 520000, 522000, 524000, 526000, 528000, 530000, 532000, 534000, 536000, 538000, 540000, 542000, 544000, 546000, 548000, 550000, 552000, 554000, 556000, 558000, 560000, 562000, 564000, 566000, 568000, 570000, 572000, 574000, 576000, 578000, 580000, 582000, 584000, 586000, 588000, 590000, 592000, 594000, 596000, 598000,600000, 602000, 604000, 606000, 608000, 610000, 612000, 614000, 616000, 618000, 620000, 622000, 624000, 626000, 628000, 630000, 632000, 634000, 636000, 638000, 640000, 642000, 644000, 646000, 648000, 650000, 652000, 654000, 656000, 658000, 660000, 662000, 664000, 666000, 668000, 670000, 672000, 674000, 676000, 678000, 680000, 682000, 684000, 686000, 688000, 690000, 692000, 694000, 696000, 698000, 700000, 702000, 704000, 706000, 708000, 710000, 712000, 714000, 716000, 718000, 720000, 722000, 724000, 726000, 728000, 730000, 732000, 734000, 736000, 738000, 740000, 742000, 744000, 746000, 748000, 750000, 752000, 754000, 756000, 758000, 760000, 762000, 764000, 766000, 768000, 770000, 772000, 774000, 776000, 778000, 780000, 782000, 784000, 786000, 788000, 790000, 792000, 794000, 796000, 798000, 800000, 802000, 804000, 806000, 808000, 810000, 812000, 814000, 816000, 818000, 820000, 822000, 824000, 826000, 828000, 830000, 832000, 834000, 836000, 838000, 840000, 842000, 844000, 846000, 848000, 850000, 852000, 854000, 856000, 858000, 860000, 862000, 864000, 866000, 868000, 870000, 872000, 874000, 876000, 878000, 880000, 882000, 884000, 886000, 888000, 890000, 892000, 894000, 896000, 898000,900000, 902000, 904000, 906000, 908000, 910000, 912000, 914000, 916000, 918000, 920000, 922000, 924000, 926000, 928000, 930000, 932000, 934000, 936000, 938000, 940000, 942000, 944000, 946000, 948000, 950000, 952000, 954000, 956000, 958000, 960000, 962000, 964000, 966000, 968000, 970000, 972000, 974000, 976000, 978000, 980000, 982000, 984000, 986000, 988000, 990000, 992000, 994000, 996000, 998000, 1000000, 1002000, 1004000, 1006000, 1008000, 1010000, 1012000, 1014000, 1016000, 1018000, 1020000, 1022000, 1024000, 1026000, 1028000, 1030000, 1032000, 1034000, 1036000, 1038000, 1040000, 1042000, 1044000, 1046000, 1048000, 1050000, 1052000, 1054000, 1056000, 1058000, 1060000, 1062000, 1064000, 1066000, 1068000, 1070000, 1072000, 1074000, 1076000, 1078000, 1080000, 1082000, 1084000, 1086000, 1088000, 1090000, 1092000, 1094000, 1096000, 1098000, 1100000, 1102000, 1104000, 1106000, 1108000, 1110000, 1112000, 1114000, 1116000, 1118000, 1120000, 1122000, 1124000, 1126000, 1128000, 1130000, 1132000, 1134000, 1136000, 1138000, 1140000, 1142000, 1144000, 1146000, 1148000, 1150000, 1152000, 1154000, 1156000, 1158000, 1160000, 1162000, 1164000, 1166000, 1168000, 1170000, 1172000, 1174000, 1176000, 1178000, 1180000, 1182000, 1184000, 1186000, 1188000, 1190000, 1192000, 1194000, 1196000, 1198000, 1200000, 1202000, 1204000, 1206000, 1208000, 1210000, 1212000, 1214000, 1216000, 1218000, 1220000, 1222000, 1224000, 1226000, 1228000, 1230000, 1232000, 1234000, 1236000, 1238000, 1240000, 1242000, 1244000, 1246000, 1248000, 1250000, 1252000, 1254000, 1256000, 1258000, 1260000, 1262000, 1264000, 1266000, 1268000, 1270000, 1272000, 1274000, 1276000, 1278000, 1280000, 1282000, 1284000, 1286000, 1288000, 1290000, 1292000, 1294000, 1296000, 1298000, 1300000, 1302000, 1304000, 1306000, 1308000, 1310000, 1312000, 1314000, 1316000, 1318000, 1320000, 1322000, 1324000, 1326000, 1328000, 1330000, 1332000, 1334000, 1336000, 1338000, 1340000, 1342000, 1344000, 1346000, 1348000, 1350000, 1352000, 1354000, 1356000, 1358000, 1360000, 1362000, 1364000, 1366000, 1368000, 1370000, 1372000, 1374000, 1376000, 1378000, 1380000, 1382000, 1384000, 1386000, 1388000, 1390000, 1392000, 1394000, 1396000, 1398000, 1400000, 1402000, 1404000, 1406000, 1408000, 1410000, 1412000, 1414000, 1416000, 1418000, 1420000, 1422000, 1424000, 1426000, 1428000, 1430000, 1432000, 1434000, 1436000, 1438000, 1440000, 1442000, 1444000, 1446000, 1448000, 1450000, 1452000, 1454000, 1456000, 1458000, 1460000, 1462000, 1464000, 1466000, 1468000, 1470000, 1472000, 1474000, 1476000, 1478000, 1480000, 1482000, 1484000, 1486000, 1488000, 1490000, 1492000, 1494000, 1496000, 1498000, 1500000, 1502000, 1504000, 1506000, 1508000, 1510000, 1512000, 1514000, 1516000, 1518000, 1520000, 1522000, 1524000, 1526000, 1528000, 1530000, 1532000, 1534000, 1536000, 1538000, 1540000, 1542000, 1544000, 1546000, 1548000, 1550000, 1552000, 1554000, 1556000, 1558000, 1560000, 1562000, 1564000, 1566000, 1568000, 1570000, 1572000, 1574000, 1576000, 1578000, 1580000, 1582000, 1584000, 1586000, 1588000, 1590000, 1592000, 1594000, 1596000, 1598000, 1600000, 1602000, 1604000, 1606000, 1608000, 1610000, 1612000, 1614000, 1616000, 1618000, 1620000, 1622000, 1624000, 1626000, 1628000, 1630000, 1632000, 1634000, 1636000, 1638000, 1640000, 1642000, 1644000, 1646000, 1648000, 1650000, 1652000, 1654000, 1656000, 1658000, 1660000, 1662000, 1664000, 1666000, 1668000, 1670000, 1672000, 1674000, 1676000, 1678000, 1680000, 1682000, 1684000, 1686000, 1688000, 1690000, 1692000, 1694000, 1696000, 1698000, 1700000, 1702000, 1704000, 1706000, 1708000, 1710000, 1712000, 1714000, 1716000, 1718000, 1720000, 1722000, 1724000, 1726000, 1728000, 1730000, 1732000, 1734000, 1736000, 1738000, 1740000, 1742000, 1744000, 1746000, 1748000, 1750000, 1752000, 1754000, 1756000, 1758000, 1760000, 1762000, 1764000, 1766000, 1768000, 1770000, 1772000, 1774000, 1776000, 1778000, 1780000, 1782000, 1784000, 1786000, 1788000, 1790000, 1792000, 1794000, 1796000, 1798000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
