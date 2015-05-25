from .graph_crf import GraphCRF

class HighOrderCRF(GraphCRF):
    """HighOrder CRF is not a general crf model, it is designed to apply to ME recognition and FC recognition.

    High order CRF consists of two potential functions: unary potential functions and pairwise potential functions.
    You can define arbitrary number of orders as long as you know how they should assemble in high order potential function.

    For example, you have a potential function \phi(X_i, Y_i)\phi(X_i, X_j, Y_i, Y_j, W_<i,j>), 
    Y_i takes 100 labels, Y_j is the same, but W only takes two values. Then labels=[100,2], high_order_parameter=[0,0,1]


    """
    def __init__(self, labels, high_order_parameter=[0], n_unary_features=None, n_edge_features=None, 
        inference_method=None, class_weight=None):
        if max(high_order_parameter)+1!=len(labels):
            raise ValueError("Parameter index specified in high order function does not corresponds to given label \
                numbers.")
        

