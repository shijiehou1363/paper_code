import sys
import ctypes
import numpy as np
import random
from ctypes import CDLL, c_int, c_double, c_void_p


class RTree:
    def __init__(self, max_entry, min_entry_factor):
        self.lib = CDLL('rtree_cppout/mytree')

        # needed C funcs
        self.lib.ConstructTree.argtypes = [c_int, c_int]
        self.lib.ConstructTree.restype = c_void_p

        self.lib.SetDefaultInsertStrategy.argtypes = [c_void_p, c_int]
        self.lib.SetDefaultInsertStrategy.restype = c_void_p

        self.lib.SetDefaultSplitStrategy.argtypes = [c_void_p, c_int]
        self.lib.SetDefaultSplitStrategy.restype = c_void_p

        self.lib.InsertRect.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib.InsertRect.restype = c_void_p

        self.lib.GetRoot.argtypes = [c_void_p]
        self.lib.GetRoot.restype = c_void_p

        self.lib.DefaultInsert.argtypes = [c_void_p, c_void_p]
        self.lib.DefaultInsert.restype = c_void_p

        self.lib.DefaultSplit.argtypes = [c_void_p, c_void_p]
        self.lib.DefaultSplit.restype = c_void_p

        self.lib.Clear.argtypes = [c_void_p]
        self.lib.Clear.restype = c_void_p

        self.lib.IsLeaf.argtypes = [c_void_p]
        self.lib.IsLeaf.restype = c_int

        self.lib.IsRoot.argtypes = [c_void_p]
        self.lib.IsRoot.restype = c_int

        self.lib.NodeEntries.argtypes = [c_void_p]
        self.lib.NodeEntries.restype = c_int

        self.lib.NodeID.argtypes = [c_void_p]
        self.lib.NodeID.restype = c_int

        self.lib.GetMinAreaContainingChild.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.GetMinAreaContainingChild.restype = c_int

        self.lib.InsertWithLoc.argtypes = [c_void_p, c_void_p, c_int, c_void_p]
        self.lib.InsertWithLoc.restype = c_void_p

        self.lib.QueryRectangle.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib.QueryRectangle.restype = c_int

        self.lib.TreeHeight.argtypes = [c_void_p]
        self.lib.TreeHeight.restype = c_int

        self.lib.CopyTree.argtypes = [c_void_p, c_void_p]
        self.lib.CopyTree.restype = c_void_p

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.lib.MakeRect.argtypes = [c_double, c_double, c_double, c_double]
        self.lib.MakeRect.restype = c_void_p

        self.lib.TraverseTree.argtypes = [c_void_p]
        self.lib.TraverseTree.restype = c_void_p

        self.lib.DirectInsert.argtypes = [c_void_p, c_void_p]
        self.lib.DirectInsert.restype = c_void_p

        self.lib.DirectSplitWithReinsert.argtypes = [c_void_p, c_void_p]
        self.lib.DirectSplitWithReinsert.restype = c_void_p

        self.lib.InsertWithEvaluatedLoc.argtypes = [c_void_p, c_void_p, c_int, c_void_p]
        self.lib.InsertWithEvaluatedLoc.restype = c_void_p

        self.lib.RetrieveEvaluatedInsertStatesByType.argtypes = [c_void_p, c_void_p, c_void_p, c_int, ctypes.POINTER(c_double), c_int, c_int]
        self.lib.RetrieveEvaluatedInsertStatesByType.restype = c_void_p

        # attrs
        
        self.strategy_map = {
            "INS_AREA":0, "INS_MARGIN":1, "INS_OVERLAP":2, "INS_RSTAR":3, 
            "SPL_MIN_AREA":0, "SPL_MIN_MARGIN":1, "SPL_MIN_OVERLAP":2, "SPL_QUADRATIC":3
        }
        self.insert_strategy = None
        self.split_strategy = None
        self.max_entry = max_entry
        self.min_entry_factor = min_entry_factor
        self.tree_ptr = self.lib.ConstructTree(int(max_entry), int(self.max_entry * self.min_entry_factor))

        # attrs set at other place
        """
        self.rec_ptr
        self.node_ptr
        self.next_node_ptr
        """

    def SetDefaultInsertStrategy(self, strategy):
        self.insert_strategy = strategy
        self.lib.SetDefaultInsertStrategy(self.tree_ptr, self.strategy_map[strategy])
    
    def SetDefaultSplitStrategy(self, strategy):
        self.split_strategy = strategy
        self.lib.SetDefaultSplitStrategy(self.tree_ptr, self.strategy_map[strategy])
    
    def PrepareRectangle(self, ll_x, ll_y, tr_x, tr_y):
        """ 
        Insert incoming rectangle data into tree object buffer

        get the rectangle and tree root node pointer
        """
        self.rec_ptr = self.lib.InsertRect(self.tree_ptr, ll_x, ll_y, tr_x, tr_y)
        self.node_ptr = self.lib.GetRoot(self.tree_ptr)

    def DefaultInsert(self, ll_x, ll_y, tr_x, tr_y):
        self.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
        self.lib.DefaultInsert(self.tree_ptr, self.rec_ptr)
    
    def DefaultSplit(self):
        self.lib.DefaultSplit(self.tree_ptr, self.node_ptr)
    
    def Clear(self):
        self.lib.Clear(self.tree_ptr)

    def GetMinAreaContainingChild(self):
        if self.lib.IsLeaf(self.node_ptr):
            return None
        child = self.lib.GetMinAreaContainingChild(self.tree_ptr, self.node_ptr, self.rec_ptr)
        if child < 0:
            return None
        else:
            # print(f"\nchild_id: {child}\n")
            return child
        
    def InsertWithLoc(self, loc):
        self.next_node_ptr = self.lib.InsertWithLoc(self.tree_ptr, self.node_ptr, loc, self.rec_ptr)
        # if current node is leaf, get the terminal state
        # else circularly update current node-ptr to the next node-ptr 
        # to enter next level to choose child to insert rectangle
        if self.lib.IsLeaf(self.node_ptr):
            return True
        else:
            self.node_ptr = self.next_node_ptr
            return False

    def Query(self, boundary):
        node_access = self.lib.QueryRectangle(self.tree_ptr, boundary[0], boundary[1], boundary[2], boundary[3])
        return node_access

    def AccessRate(self, boundary):
        node_access = self.lib.QueryRectangle(self.tree_ptr, boundary[0], boundary[1], boundary[2], boundary[3])
        height = self.lib.TreeHeight(self.tree_ptr)
        if height == 0:
            print("height is 0")
            input()
        return 1.0 * node_access / height
    
    def CopyTree(self, tree):
        self.lib.CopyTree(self.tree_ptr, tree)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def GetTreeHeight(self):
        return self.lib.TreeHeight(self.tree_ptr)
    
    def GetRoot(self):
        return self.lib.GetRoot(self.tree_ptr)

    def MakeRect(self, ll_x, ll_y, tr_x, tr_y):
        return self.lib.MakeRect(ll_x, ll_y, tr_x, tr_y)

    def IsRoot(self, node):
        return self.lib.IsRoot(node)
    
    def NodeEntries(self, node):
        return self.lib.NodeEntries(node)

    def NodeID(self, node):
        return self.lib.NodeID(node)
    
    def IsLeaf(self, node):
        return self.lib.IsLeaf(node)
    
    def TraverseTreeInfo(self):
        self.lib.TraverseTree(self.tree_ptr)   

    def DirectInsert(self, ll_x, ll_y, tr_x, tr_y):
        self.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
        self.node_ptr = self.lib.DirectInsert(self.tree_ptr, self.rec_ptr)

    def DirectSplitWithReinsert(self):
        self.lib.DirectSplitWithReinsert(self.tree_ptr, self.node_ptr)

    def InsertWithEvaluatedLoc(self, action):
        self.next_node_ptr = self.lib.InsertWithEvaluatedLoc(self.tree_ptr, self.node_ptr, action, self.rec_ptr)
        if self.lib.IsLeaf(self.node_ptr):
            return True
        else:
            self.node_ptr = self.next_node_ptr
            return False

    def RetrieveEvaluatedInsertStatesByType(self, action_space, num_features, feature_type):
        if self.lib.IsLeaf(self.node_ptr):
            return None
        state_length = num_features * action_space         # action_space is topk, action_spcae_size
        state_c = (c_double * state_length)()
        self.lib.RetrieveEvaluatedInsertStatesByType(self.tree_ptr, self.node_ptr, self.rec_ptr, action_space, state_c, num_features, feature_type)
        states = np.ctypeslib.as_array(state_c)
        return states

