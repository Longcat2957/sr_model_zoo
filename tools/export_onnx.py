import torch.onnx

def export_onnx_model(model, input_shape, onnx_file_path):
    """
    Export PyTorch model to ONNX format
    Args:
        model: PyTorch model to be exported
        input_shape: tuple representing the input shape of the model
        onnx_file_path: file path to save the exported ONNX model
    """
    # Set the model to inference mode
    model.eval()

    # Create a dummy input with the specified shape
    dummy_input = torch.randn(input_shape)

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=["input"], output_names=["output"])