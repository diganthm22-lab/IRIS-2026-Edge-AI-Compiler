import onnx
import numpy as np

onnx_path = '/Users/drkrm/Documents/IRIS/victim_model.onnx'

def calculate_tensor_size(shape, data_type=onnx.TensorProto.FLOAT):
    """Calculates memory size of a tensor in bytes based on its shape."""
    if not shape:
        return 0

    elements = 1
    for dim in shape:
        if isinstance(dim, int):
            elements *= dim
        elif hasattr(dim, 'dim_value') and dim.dim_value > 0:
            elements *= dim.dim_value
            
 
    bytes_per_element = 4 
    return elements * bytes_per_element

def analyze_onnx_graph():
    # Load the ONNX model
    model = onnx.load(onnx_path)
    graph = model.graph
    
    print("==================================================")
    print(" IRIS MEMORY COMPILER: INITIAL GRAPH ANALYSIS ")
    print("==================================================")
    
   
    print("\n [STATIC MEMORY] Analyzing Model Weights:")
    total_weight_bytes = 0
    for initializer in graph.initializer:
        shape = list(initializer.dims)
        size_bytes = calculate_tensor_size(shape)
        total_weight_bytes += size_bytes
        print(f"  • Weight: '{initializer.name}' | Shape: {shape} | Size: {size_bytes} Bytes")
        
   
    print("\n [TRANSIENT MEMORY] Analyzing Computational Flow:")
    
    
    tensor_lifespans = {}
    
    for idx, node in enumerate(graph.node):
        print(f"  Node #{idx+1}: OpType='{node.op_type}' | Name='{node.name}'")
        print(f"    Inputs:  {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
        
       
        for t_input in node.input:
            if t_input not in tensor_lifespans:
                tensor_lifespans[t_input] = {'birth': idx, 'death': idx}
            else:
                tensor_lifespans[t_input]['death'] = idx
                
        for t_output in node.output:
            if t_output not in tensor_lifespans:
                tensor_lifespans[t_output] = {'birth': idx, 'death': idx}
            else:
                tensor_lifespans[t_output]['death'] = idx

    print("\n Summary Totals:")
    print(f"  • Total Static Weight Memory: {total_weight_bytes} Bytes ({total_weight_bytes/1024:.2f} KB)")
    print(f"  • Total Tracked Intermediate Tensors: {len(tensor_lifespans)}")
    print("==================================================")

if __name__ == "__main__":
    analyze_onnx_graph()
