import numpy as np
import opengm

from .utils import loss_augment_unaries

class EdgeLabelCRF(object):
    def __init__(self, nNodeLabel, nEdgeLabel=2, 
                 nNodeFeature=None, nEdgeFeature=None, nInteractFeature=None,
                 inference_method=None, class_weight=None):
        self.nNodeLabel=nNodeLabel
        self.nEdgeLabel=nEdgeLabel
        self.nNodeFeature=nNodeFeature
        self.nEdgeFeature=nEdgeFeature
        self.nInteractFeature=nInteractFeature
        if inference_method is None:
            self.inference_method=('ogm',{'alg':'fm'})
        else:
            self.inference_method=inference_method

        self.class_weight=class_weight
        self.inference_calls=0

    def __repr__(self):
        return ("%s(With %d node labels, %d edge labels, %d node features, %d edge features, \
            %d interact features, inference method %s)"%(type(self).__name__, self.nNodeLabel, \
            self.nEdgeLabel, self.nNodeFeature, self.nEdgeFeature, self.nInteractFeature, self.inference_method))

    def _get_node_potentials(self, x, w):
        """x here is observation. Model doesn't need to know real observation, such as how many nodes a graph have.
        x=(node_feature, edge_feature, inter_feature, edges_list)

        Here we concern about node_feature. Node_feature is shaped like nNodes x nNodeFeature.
        """
        node_feature=x[0]
        if node_feature.shape[1]!=self.nNodeFeature:
            raise ValueError("Number of node features not match the model.")
        param = w[:self.nNodeFeature*self.nNodeLabel].reshape(self.nNodeFeature, self.nNodeLabel)
        return np.dot(node_feature, param)

    def _get_edge_potentials(self, x, w):
        """Here we concern about edge_feature, which is isolated feature. nEdges x nEdgeFeature
        """
        edge_feature=x[1]
        if edge_feature.shape[1]!=self.nEdgeFeature:
            raise ValueError("Number of edge features not match the model.")
        param = w[self.nNodeFeature*self.nNodeLabel:self.nNodeFeature*self.nNodeLabel+self.nEdgeLabel*self.nEdgeFeature].\
        reshape(self.nEdgeFeature, self.nEdgeLabel)
        return np.dot(edge_feature, param)

    def _get_inter_potentials(self, x, w):
        """Here we concern about the inter_feature, which is nEdges x nInteractFeature
        """
        inter_feature=x[2]
        if inter_feature.shape[1]!=self.nInteractFeature:
            raise ValueError("Number of interact features not match the model.")
        param = w[self.nNodeFeature*self.nNodeLabel+self.nEdgeLabel*self.nEdgeFeature:].reshape(self.nInteractFeature,-1)
        inter_potentials = np.dot(inter_feature, param).reshape(inter_feature.shape[0], \
            self.nNodeLabel, self.nNodeLabel, self.nEdgeLabel)
        for i in xrange(inter_feature.shape[0]):#This may introduce some incompatiblity because we assume the nEdgeFeature is 2.
            inter_potentials[i,:,:,1]*=np.diag(np.ones(self.nNodeLabel))
        return inter_potentials

    def _check_size_w(self, w):
        if w.shape[0]!=self.nNodeLabel*self.nNodeFeature+self.nEdgeLabel*self.nEdgeFeature+\
        self.nInteractFeature*self.nEdgeFeature*self.nNodeFeature*self.nNodeFeature:
            raise ValueError("Number of w mismatch model.")

    def _set_size_joint_feature(self):
        if self.nNodeFeature is not None and self.nEdgeFeature is not None and self.nInteractFeature is not None:
            self.size_joint_feature = self.nNodeLabel*self.nNodeFeature + self.nEdgeLabel*self.nEdgeFeature + \
            self.nInteractFeature*self.nEdgeLabel*self.nNodeLabel*self.nNodeLabel


    def inference(self, x, w, relaxed=True, return_energy=False):
        self._check_size_w(w)
        self.inference_calls += 1
        node_potentials = self._get_node_potentials(x, w)
        edge_potentials = self._get_edge_potentials(x, w)
        inter_potentials = self._get_inter_potentials(x, w)
        edges = x[3]

        return self._inference(node_potentials, edge_potentials, inter_potentials, edges, self.inference_method[1]['alg'], \
            return_energy=return_energy)

    def batch_inference(self, X, w, relaxed=None):
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def _inference(self, node_potentials, edge_potentials, inter_potentials, 
                  edges, alg, return_energy=False, init=None, **kwargs):
        if node_potentials.shape[1]!=self.nNodeLabel:
            raise ValueError("Node feature function parameters should match node label numbers.")
        if edge_potentials.shape[0]!=edges.shape[0]:
            raise ValueError("Edge feature function numbers should match given edges.")
        if edge_potentials.shape[1]!=self.nEdgeLabel:
            raise ValueError("Edge feature function parameters should match edge label numbers.")
        if inter_potentials.shape[0]!=edges.shape[0]:
            raise ValueError("Interaction potential function number should match edge numbers.")
        if inter_potentials.shape[1]!=self.nNodeLabel or inter_potentials.shape[2]!=self.nNodeLabel \
        or inter_potentials.shape[3]!=self.nEdgeLabel:
            raise ValueError("Interaction potential function parameters should match combination number of labels.")

        nNodes = node_potentials.shape[0]
        nEdges = edges.shape[0]

        gm = opengm.gm(np.hstack([np.ones(nNodes, dtype=opengm.label_type)*self.nNodeLabel, \
            np.ones(nEdges, dtype=opengm.label_type)*self.nEdgeLabel]))
        gm.reserveFactors(nNodes+2*nEdges)
        gm.reserveFunctions(nNodes+2*nEdges,'explicit')

        unaries = -np.require(node_potentials, dtype=opengm.value_type)
        fidUnaries = gm.addFunctions(unaries)
        visUnaries = np.arange(nNodes, dtype=np.uint64)
        gm.addFactors(fidUnaries, visUnaries)

        unaries = -np.require(edge_potentials, dtype=opengm.value_type)
        fidUnaries = gm.addFunctions(unaries)
        visUnaries = np.arange(nNodes, nEdges+nNodes, dtype=np.uint64)
        gm.addFactors(fidUnaries, visUnaries)

        highOrderFunctions = -np.require(inter_potentials, dtype=opengm.value_type)
        fidHighOrder = gm.addFunctions(highOrderFunctions)
        vidHighOrder = np.hstack([edges, np.arange(nNodes, nNodes+nEdges).reshape((nEdges,1))])
        vidHighOrder = np.require(vidHighOrder, dtype=np.uint64)
        gm.addFactors(fidHighOrder, vidHighOrder)

        if alg == 'bp':
            inference = opengm.inference.BeliefPropagation(gm)
        elif alg == 'dd':
            inference = opengm.inference.DualDecompositionSubgradient(gm)
        elif alg == 'trws':
            inference = opengm.inference.TrwsExternal(gm)
        elif alg == 'trw':
            inference = opengm.inference.TreeReweightedBp(gm)
        elif alg == 'gibbs':
            inference = opengm.inference.Gibbs(gm)
        elif alg == 'lf':
            inference = opengm.inference.LazyFlipper(gm)
        elif alg == 'icm':
            inference = opengm.inference.Icm(gm)
        elif alg == 'dyn':
            inference = opengm.inference.DynamicProgramming(gm)
        elif alg == 'fm':
            inference = opengm.inference.AlphaExpansionFusion(gm)
        elif alg == 'gc':
            inference = opengm.inference.GraphCut(gm)
        elif alg == 'loc':
            inference = opengm.inference.Loc(gm)
        elif alg == 'mqpbo':
            inference = opengm.inference.Mqpbo(gm)
        elif alg == 'alphaexp':
            inference = opengm.inference.AlphaExpansion(gm)
        if init is not None:
            inference.setStartingPoint(init)

        inference.infer()
        res = inference.arg().astype(np.int)
        if return_energy:
            return res, gm.evaluate(res)
        return res

    def joint_feature(self, x, y):
        node_feature = x[0]
        edge_feature = x[1]
        inter_feature = x[2]
        edges = x[3]
        node_label = y[:node_feature.shape[0]]
        edge_label = y[node_feature.shape[0]:]

        node_marginals = np.zeros((self.nNodeLabel, node_feature.shape[0]), dtype=np.int)
        node_marginals[node_label, np.arange(node_feature.shape[0])]=1
        node_acc = np.dot(node_marginals, node_feature).T

        edge_marginals = np.zeros((self.nEdgeLabel, edge_feature.shape[0]), dtype=np.int)
        edge_marginals[edge_label, np.arange(edge_feature.shape[0])]=1
        edge_acc = np.dot(edge_marginals, edge_feature).T

        inter_marginals = np.zeros((self.nEdgeLabel*self.nNodeLabel*self.nNodeLabel,\
            inter_feature.shape[0]), dtype=np.int)
        t=edge_label + (node_label[edges[:,1]] + node_label[edges[:,0]]*\
            self.nNodeLabel)*self.nEdgeLabel
        inter_marginals[t, np.arange(inter_feature.shape[0])]=1
        inter_acc = np.dot(inter_marginals, inter_feature).T

        return np.hstack([node_acc.ravel(), edge_acc.ravel(), inter_acc.ravel()])

    def loss(self, y, y_hat):
        return np.sum(y != y_hat)

    def loss_augmented_inference(self, x, y, w, relaxed=False, return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        node_potentials = self._get_node_potentials(x, w)
        edge_potentials = self._get_edge_potentials(x, w)
        inter_potentials = self._get_inter_potentials(x, w)
        edges = x[3]
        loss_augment_unaries(node_potentials, np.asarray(y[:node_potentials.shape[0]]), np.ones(self.nNodeLabel))
        loss_augment_unaries(edge_potentials, np.asarray(y[node_potentials.shape[0]:]), np.ones(self.nEdgeLabel))

        return self._inference(node_potentials, edge_potentials, inter_potentials, edges, self.inference_method[1]['alg'], \
            return_energy=return_energy, relaxed=relaxed)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]

    def initialize(self, X, Y):
        nNodeFeature = X[0][0].shape[1]
        if self.nNodeFeature is None:
            self.nNodeFeature = nNodeFeature
        elif self.nNodeFeature != nNodeFeature:
            raise ValueError("Expected %d features, got %d"
                             % (self.nNodeFeature, nNodeFeature))

        nEdgeFeature = X[0][1].shape[1]
        if self.nEdgeFeature is None:
            self.nEdgeFeature = nEdgeFeature
        elif self.nEdgeFeature != nEdgeFeature:
            raise ValueError("Expected %d features, got %d"
                             % (self.nEdgeFeature, nEdgeFeature))

        nInteractFeature = X[0][2].shape[1]
        if self.nInteractFeature is None:
            self.nInteractFeature = nInteractFeature
        elif self.nInteractFeature != nInteractFeature:
            raise ValueError("Expected %d edge features, got %d"
                             % (self.nInteractFeature, nInteractFeature))

        self._set_size_joint_feature()