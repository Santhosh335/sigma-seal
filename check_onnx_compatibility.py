import onnx

model = onnx.load("sigma_seal_model.onnx")
print("IR version:", model.ir_version)
print("Opset version:", model.opset_import[0].version)
print("\nOp types used in the graph:")
for node in model.graph.node:
    print(" -", node.op_type)