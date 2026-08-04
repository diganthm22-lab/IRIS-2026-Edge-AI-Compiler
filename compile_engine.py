import onnx
from onnx import numpy_helper
import numpy as np

onnx_path = '/Users/drkrm/Documents/IRIS/victim_model.onnx'
cpp_header_path = '/Users/drkrm/Documents/IRIS/deployed_model.h'

def calculate_tensor_bytes(shape, is_quantized=True):
    if not shape or len(shape) == 0:
        return 4
    elements = 1
    for dim in shape:
        if isinstance(dim, int):
            elements *= dim
        elif hasattr(dim, 'dim_value') and dim.dim_value > 0:
            elements *= dim.dim_value
    return elements * (1 if is_quantized else 4)

def run_complete_compiler_with_export():
    print("==========================================================")
    print("   IRIS MEMORY COMPILER: GENERATING C++ DEPLOYMENT    ")
    print("==========================================================")
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    total_original_weights = 2948
    total_quantized_weights = 737
    
    cpp_weight_definitions = []
    
    for initializer in graph.initializer:
        weight_data = numpy_helper.to_array(initializer)
      
        max_val = np.max(np.abs(weight_data))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized_data = np.round(weight_data / scale).astype(np.int8).flatten()
        
        array_vals = ", ".join(map(str, quantized_data))
        cpp_weight_definitions.append(f"// Scale Factor: {scale:.6f}\nconst int8_t {initializer.name}_quantized[{len(quantized_data)}] = {{{array_vals}}};")

    activation_shapes = {
        'sensor_input': [1, 5], 'linear': [1, 32], 'relu': [1, 32],
        'linear_1': [1, 16], 'relu_1': [1, 16], 'linear_2': [1, 1], 'anomaly_output': [1, 1]
    }
    
    tensor_lifespans = {}
    for idx, node in enumerate(graph.node):
        for t_input in node.input:
            if t_input in activation_shapes:
                if t_input not in tensor_lifespans: tensor_lifespans[t_input] = {'birth': idx, 'death': idx}
                else: tensor_lifespans[t_input]['death'] = idx
        for t_output in node.output:
            if t_output in activation_shapes:
                if t_output not in tensor_lifespans: tensor_lifespans[t_output] = {'birth': idx, 'death': idx}
                else: tensor_lifespans[t_output]['death'] = idx

    naive_activation_memory = sum([calculate_tensor_bytes(activation_shapes[n], True) for n in tensor_lifespans])
    peak_arena_memory = 0
    
    for step in range(len(graph.node)):
        active_bytes_at_step = sum([calculate_tensor_bytes(activation_shapes[name], True) for name, info in tensor_lifespans.items() if info['birth'] <= step <= info['death']])
        if active_bytes_at_step > peak_arena_memory:
            peak_arena_memory = active_bytes_at_step

    with open(cpp_header_path, 'w') as f:

        f.write("#ifndef DEPLOYED_MODEL_H\n")
        f.write("#define DEPLOYED_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// --- STATIC COMPRESSED MODEL WEIGHTS ---\n")
        for weight_def in cpp_weight_definitions:
            f.write(weight_def + "\n\n")
        f.write("// --- STATIC TENSOR ARENA MEMORY CONFIGURATION ---\n")
        f.write(f"const uint32_t TOTAL_WEIGHT_BYTES = {total_quantized_weights};\n")
        f.write(f"const uint32_t TENSOR_ARENA_SIZE_BYTES = {peak_arena_memory};\n\n")
        f.write("// Allocate the static, shared memory pool directly in the microcontroller's RAM\n")
        f.write("uint8_t tensor_arena[TENSOR_ARENA_SIZE_BYTES];\n\n")
        f.write("#endif // DEPLOYED_MODEL_H\n")    
    print(f" Exported to: {cpp_header_path}")
    print("==========================================================")

if __name__ == "__main__":
    run_complete_compiler_with_export()
