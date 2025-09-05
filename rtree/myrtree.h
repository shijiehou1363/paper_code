#ifndef MYRTREE_H
#define MYRTREE_H

#include <utility>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>
#include <assert.h>
#include <limits>
#include <list>
#include <vector>
#include <cmath>
#include <queue>
#include <numeric>
#include <iterator>


enum INSERT_STRATEGY{
    INS_AREA, INS_MARGIN, INS_OVERLAP, INS_RSTAR
};

enum SPLIT_STRATEGY{
    SPL_MIN_AREA, SPL_MIN_MARGIN, SPL_MIN_OVERLAP, SPL_QUADRATIC
};

std::string GetInsertStrategyName(INSERT_STRATEGY strategy);
std::string GetSplitStrategyName(SPLIT_STRATEGY strategy);
INSERT_STRATEGY GetInsertStrategyByInt(int value);
SPLIT_STRATEGY GetSplitStrategyByInt(int value);

struct Stats {
	int node_access;
	int action_history[5];
	void Reset(); 
	void TakeAction(int action);
};

struct Point {
public:
    std::pair<double, double> data;
public:
    Point() : data(0.0, 0.0) {}
    Point(double x, double y) : data(x, y) {}
    double get_x() const { return data.first; }
    double get_y() const { return data.second; }
    bool operator==(const Point& point) const { return data == point.data; }

    friend std::ostream& operator<<(std::ostream& os, const Point& point);
};

// =================================================================================================================================================================================================

class Rectangle {
public:
    // directly get rectangle info
    Point low_left;
    Point top_right;
    unsigned int id;

public:
    // constructor
    Rectangle() : low_left(0.0, 0.0), top_right(0.0, 0.0) {};
    Rectangle(Point point1, Point point2);
    Rectangle(const Rectangle& rectangle) : low_left(rectangle.low_left), top_right(rectangle.top_right) {};
    Rectangle(double ll_x, double ll_y, double tr_x, double tr_y);
    bool operator==(const Rectangle& rectangle) const { return low_left == rectangle.low_left && top_right == rectangle.top_right; }

    friend std::ostream& operator<<(std::ostream& os, const Rectangle& rectangle); 

    // directly get rectangle features
    double Left() const { return low_left.get_x(); }
    double Bottom() const { return low_left.get_y(); }
    double Top() const { return top_right.get_y(); }
    double Right() const { return top_right.get_x(); }
    Point Center() const { return Point( (low_left.get_x() + top_right.get_x()) / 2, (low_left.get_y() + top_right.get_y()) / 2); }
    double CenterDis(const Rectangle* rec);
    double Area() const { return (this->Right() - this->Left()) * (this->Top() - this->Bottom()); }
    double Perimeter() const { return 2 * (this->Right() - this->Left() + this->Top() - this->Bottom()); }

    // changes of rectangle
    void SetAs(const Rectangle& rectangle);
    void SetAs(double ll_x, double ll_y, double tr_x, double tr_y);
    void Include(const Rectangle& rectangle);
    
    // relationships between rectangles
    bool Contains(const Rectangle* rec);
    bool IsOverlap(const Rectangle* rec);

    Rectangle GetUnion(const Rectangle& rectangle);
    double GetOverlapArea(const Rectangle& rec);
};

class RTreeNode : public Rectangle {
public:
    int parent;
    std::vector<int> children;       // records id of children
    
    int entry_num;
    int level;
    /* when the attr level will be update?

        1. Anytime initialize a new node, level is 0 of copy other node's level
        2. root spliting, which creates a new root, new root-level+1 and tree-height+1
    */
    bool is_overflow;
    bool is_leaf;
    /* when the attr is_leaf will be update?

        1. Initialize a RTree, set new root's is_leaf true
        2. root splits, set root's is_leaf false
        3. leaf or internal node splits, sibling's is_leaf set spliting node's is_leaf false

            that means:
                leaf splits, new sibling leaf's is_leaf is set true
                internal splits, new sibling internal-node's is_leaf is set false

            and tree's growing taller is from bottom, every internal node is split from the old root, which is is_leaf false
            so there's no need to care too much about node's is leaf, all depends on split
    */

    static int maxi_entries;
    static int mini_entries;
    double origin_center[2];
    static double RR_s;

public: 
    // constructor
    RTreeNode();
    RTreeNode(RTreeNode* node);

    bool AddChildren(int node_id);      // add rectangle to a leaf
    bool AddChildren(RTreeNode* node);  // add node to a internal node   

    bool CopyChildren(const std::vector<int>& nodes);

};

// =================================================================================================================================================================================================

template<class T>
Rectangle MergeRange(const std::vector<T*>& entries, const int start_idx, const int end_idx);

template<class T>
int FindMinimumSplit(const std::vector<T*>& entries, double(*score_func1)(const Rectangle &, const Rectangle &),
    double(*score_func2)(const Rectangle &, const Rectangle &), double& min_value1, double& min_value2, Rectangle& rec1, Rectangle& rec2);

double SplitArea(const Rectangle& rectangle1, const Rectangle& rectangle2);
double SplitOverlap(const Rectangle& rectangle1, const Rectangle& rectangle2);
double SplitPerimeter(const Rectangle& rectangle1, const Rectangle& rectangle2);
// =================================================================================================================================================================================================

class RTree {
public:
    double fill_factor;
    std::vector<RTreeNode*> tree_nodes_;     // all tree nodes (root, internal_node, leaf), not include rectangle data
    std::vector<Rectangle*> objects_;        // all insterted rectangle object data are recorded here

    int root_;                               // root node's id: choosed_child in tree_nodes_
    int height_;                             // root node's level + 1

    Stats stats_;

    int result_count;

    INSERT_STRATEGY insert_strategy_;
    SPLIT_STRATEGY split_strategy_;

    double RR_s;
	double RR_y1;
	double RR_ys;

    std::vector<RTreeNode*> tmp_sorted_children;
    std::vector<RTreeNode*> comprehensive_evaluated_children;

    std::vector<bool> first_overflow;

public:
    RTree();
    RTreeNode* Root();
    RTreeNode* CreateNode();
    Rectangle* InsertRectangle(double ll_x, double ll_y, double tr_x, double tr_y);

    RTreeNode* InsertStepByStep(const Rectangle* rectangle, RTreeNode* tree_node);
    RTreeNode* InsertStepByStep(const Rectangle* rectangle, RTreeNode* tree_node, INSERT_STRATEGY strategy);

    RTreeNode* SplitStepByStep(RTreeNode* tree_node);
    RTreeNode* SplitStepByStep(RTreeNode* tree_node, SPLIT_STRATEGY strategy);

    int GetMinAreaContainingChild(RTreeNode* tree_node, Rectangle* rec);
    RTreeNode* InsertInLoc(RTreeNode* tree_node, int choosed_child, Rectangle* rec);
    RTreeNode* InsertInEvaluatedLoc(RTreeNode* tree_node, int action, Rectangle* rec);

    void ComprehensiveEvaluation(RTreeNode* tree_node, Rectangle* rec);

    void GetEvaluatedInsertStates125(RTreeNode* tree_node, Rectangle* rec, double* states, int cb, int num_features);


    int Query(Rectangle& rectangle);

    void Copy(RTree* tree);

    void RetrieveForReinsert(RTreeNode* tree_node, std::list<int>& candidates);
	void UpdateMBRForReinsert(RTreeNode* tree_node);  
};

// =================================================================================================================================================================================================

extern "C" {
    RTree* ConstructTree(int maxi_entries, int mini_entries);


    void SetDefaultInsertStrategy(RTree* rtree, int strategy);
    void SetDefaultSplitStrategy(RTree* rtree, int strategy);

    Rectangle* InsertRect(RTree* rtree, double ll_x, double ll_y, double tr_x, double tr_y);

    RTreeNode* GetRoot(RTree* rtree);

    void DefaultInsert(RTree* rtree, Rectangle* rect);
    void DefaultSplit(RTree* rtree, RTreeNode* tree_node);
    void DirectSplitWithReinsert(RTree* rtree, RTreeNode* node);

    RTreeNode* DirectInsert(RTree* rtree, Rectangle* rec);
    void DirectSplit(RTree* rtree, RTreeNode* node);

    void Clear(RTree* rtree);
    void RetrieveEvaluatedInsertStatesByType(RTree* tree, RTreeNode* tree_node, Rectangle* rec, int cb, double* states, int num_features, int feature_type);
    
    int IsLeaf(RTreeNode* node);
    int IsRoot(RTreeNode* node);
    int NodeEntries(RTreeNode* node);
    int NodeID(RTreeNode* node);

    int GetMinAreaContainingChild(RTree* rtree, RTreeNode* tree_node, Rectangle* rec);

    RTreeNode* InsertWithLoc(RTree* tree, RTreeNode* tree_node, int choosed_child, Rectangle* rec);
    RTreeNode* InsertWithEvaluatedLoc(RTree* tree, RTreeNode* tree_node, int action, Rectangle* rec);

    int QueryRectangle(RTree* rtree, double ll_x, double ll_y, double tr_x, double tr_y);

    int TreeHeight(RTree* rtree);
    void CopyTree(RTree* tree, RTree* src_tree);
    void TraverseTree(RTree* rtree);

    Rectangle* MakeRect(double ll_x, double ll_y, double tr_x, double tr_y);

    
}

#endif