import os

def load_model(model_path: str, device: str = "cuda", **kwargs):
    if kwargs.get("force_ori_type", False):
        # For hubert, landmark, retinaface, mediapipe
        model = load_force_ori_type(model_path, device, **kwargs)
        return model, "ori"

    if model_path.endswith(".onnx"):
        # ONNX
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        model = onnxruntime.InferenceSession(model_path, providers=providers)
        return model, "onnx"

    elif model_path.endswith(".engine") or model_path.endswith(".trt"):
        # TensorRT
        from .tensorrt_utils import TRTWrapper

        if not os.path.isfile(model_path):
            # Try FP16 as a fallback
            alt_model_path = model_path.replace("_fp32", "_fp16")
            if os.path.isfile(alt_model_path):
                print(f"FP32 engine not found. Falling back to FP16 engine: {alt_model_path}")
                model_path = alt_model_path
            else:
                raise FileNotFoundError(
                    f"Neither FP32 nor FP16 engine file found: {model_path} or {alt_model_path}"
                )

        model = TRTWrapper(model_path)
        return model, "tensorrt"

    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        # PyTorch
        model = create_model(model_path, device, **kwargs)
        return model, "pytorch"

    else:
        raise ValueError(f"Unsupported model file type: {model_path}")


def create_model(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..models.modules",
    **kwargs,
):
    import importlib

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    model = module(**kwargs)
    model.load_model(model_path).to(device)
    return model


def load_force_ori_type(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..aux_models.modules",
    force_ori_type=False,
    **kwargs,
):
    import importlib

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    model = module(**kwargs)
    return model
