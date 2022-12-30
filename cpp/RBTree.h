#ifndef _INCLUDE_RBTREE_H_
#define _INCLUDE_RBTREE_H_

#include <stdio.h>
#include <random>

struct node {
    node(double, int);
    size_t size; int value;
    double key, sum; char color;
    node *parent, *left, *right;
};

class RBTree {
    private:
        node *root;
        void display(node *x);
        void insert_fix(node *x);
        void left_rotate(node *x);
        void right_rotate(node *x);
        void erase_fix(node *x, node *p);

        // for sampler
        std::uniform_real_distribution<double> dis;
        std::random_device rd;
        std::mt19937 gen;
        node **arr;
        double eps;
        int n;

    public:
        RBTree(int, double);
        void search();
        void display();
        int rank(node *x);
        void erase(node *p);
        int rank(double val);
        node *find(double val);
        void erase(double val);
        node *insert(double val, int idx);

        // for sampler
        int sample();
        node *solve();
        node *select_rank(int rk);
        node *select_sum(double x);
};

#endif

