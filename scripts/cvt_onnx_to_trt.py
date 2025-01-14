import os
import torch
import argparse
import platform
import ctypes
import tensorrt as trt


def load_plugin(plugin_file):
    """
    Load the custom TensorRT plugin dynamically based on the platform.
    """
    if platform.system().lower() == "linux":
        ctypes.CDLL(plugin_file, mode=ctypes.RTLD_GLOBAL)
    elif platform.system().lower() == "windows":
        ctypes.CDLL(plugin_file, mode=ctypes.RTLD_GLOBAL, winmode=0)
    else:
        raise OSError("Unsupported platform for loading plugins")


def onnx_to_trt(onnx_file, trt_file, fp16=False, more_cmd=None):
    """
    Convert an ONNX model to a TensorRT engine using the Polygraphy CLI.
    """
    cap = torch.cuda.get_device_capability()
    compatible = "--hardware-compatibility-level=Ampere_Plus" if cap[0] >= 8 else ""
    
    cmd = [
        "polygraphy",
        "convert",
        onnx_file,
        "-o",
        trt_file,
        compatible,
        "--fp16" if fp16 else "",
        f"--builder-optimization-level=5",
    ]
    
    if more_cmd:
        cmd += more_cmd

    print(" ".join(cmd))
    os.system(" ".join(cmd))


def onnx_to_trt_for_gridsample(onnx_file, trt_file, fp16=False, plugin_file="./libgrid_sample_3d_plugin.so"):
    """
    Convert an ONNX model to a TensorRT engine with a custom plugin for GridSample layers.
    """
    logger = trt.Logger(trt.Logger.INFO)
    load_plugin(plugin_file)  # Load the custom plugin
    trt.init_libnvinfer_plugins(logger, "")
    
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print(f"Failed to parse ONNX file: {onnx_file}")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(error)
            parser.clear_errors()
            return

    config = builder.create_builder_config()
    config.builder_optimization_level = 5

    # Check for hardware compatibility (Ampere+)
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8:
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

    exclude_list = [
        "SHAPE",
        "ASSERTION",
        "SHUFFLE",
        "IDENTITY",
        "CONSTANT",
        "CONCATENATION",
        "GATHER",
        "SLICE",
        "CONDITION",
        "CONDITIONAL_INPUT",
        "CONDITIONAL_OUTPUT",
        "FILL",
        "NON_ZERO",
        "ONE_HOT",
    ]

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if str(layer.type)[10:] not in exclude_list and "GridSample" in layer.name:
            print(f"Setting {layer.name} to float32 precision")
            layer.precision = trt.float32

    engine_string = builder.build_serialized_network(network, config)
    if engine_string:
        with open(trt_file, "wb") as f:
            f.write(engine_string)


def main(onnx_dir, trt_dir, grid_sample_plugin_file=""):
    """
    Main function to process ONNX models and convert them to TensorRT engines.
    """
    plugin_file = os.path.abspath(grid_sample_plugin_file)
    if not os.path.isfile(plugin_file):
        raise FileNotFoundError(f"Plugin file not found: {plugin_file}")

    names = [i[:-5] for i in os.listdir(onnx_dir) if i.endswith(".onnx")]
    for name in names:
        print("=" * 20, f"{name} start", "=" * 20)

        fp16 = False if name in {"motion_extractor", "hubert", "wavlm"} else True

        more_cmd = None
        if name == "wavlm":
            more_cmd = [
                "--trt-min-shapes audio:[1,1000]",
                "--trt-max-shapes audio:[1,320080]",
                "--trt-opt-shapes audio:[1,320080]",
            ]
        elif name == "hubert":
            more_cmd = [
                "--trt-min-shapes input_values:[1,3240]",
                "--trt-max-shapes input_values:[1,12960]",
                "--trt-opt-shapes input_values:[1,6480]",
            ]

        onnx_file = os.path.join(onnx_dir, f"{name}.onnx")
        trt_file = os.path.join(trt_dir, f"{name}_fp{16 if fp16 else 32}.engine")

        if os.path.isfile(trt_file):
            print("=" * 20, f"{name} skip", "=" * 20)
            continue

        if name == "warp_network":
            onnx_to_trt_for_gridsample(onnx_file, trt_file, fp16, plugin_file=plugin_file)
        else:
            onnx_to_trt(onnx_file, trt_file, fp16, more_cmd=more_cmd)

        print("=" * 20, f"{name} done", "=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True, help="Input ONNX directory")
    parser.add_argument("--trt_dir", type=str, required=True, help="Output TensorRT directory")
    args = parser.parse_args()

    onnx_dir = os.path.abspath(args.onnx_dir)
    trt_dir = os.path.abspath(args.trt_dir)

    assert os.path.isdir(onnx_dir), f"ONNX directory does not exist: {onnx_dir}"
    os.makedirs(trt_dir, exist_ok=True)

    plugin_filename = "grid_sample_3d_plugin.dll" if platform.system().lower() == "windows" else "libgrid_sample_3d_plugin.so"
    plugin_file = os.path.join(onnx_dir, plugin_filename)

    main(onnx_dir, trt_dir, grid_sample_plugin_file=plugin_file)
