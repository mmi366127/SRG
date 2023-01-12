#include "RBTree.h"


int size(node *x) {
    return x == nullptr ? 0 : x->size;
}

double sum(node *x) {
    return x == nullptr ? 0 : x->sum;
}

void pull(node *x) {
    if(x != nullptr) {
        x->size = 1 + size(x->left) + size(x->right);
        x->sum = x->key + sum(x->left) + sum(x->right);
    }
}

node::node(double val, int idx) {
    size = 1; value = idx;
    key = val; color = 'r'; sum = val;
    left = right = parent = nullptr;
}

RBTree::RBTree(int n_, double eps_ = 0.001) : gen(rd()), dis(0, 1), arr(n_) {
    root = nullptr; eps = eps_; n = n_;
}

node* RBTree::find(double val) {
    node *p = root;
    while(p != nullptr) {
        if(p->key == val) {
            return p;
        }
        if(p->key < val) //maybe <= ?
            p = p->right;
        else 
            p = p->left;
    }
    return nullptr;
}

bool chk_(node *x, int &sz, double &sm, double eps = 1e-6) {
    if(x == nullptr) return true;
    int _sz = 1; double _sm = x->key;
    if(!chk_(x->left, _sz, _sm)) return false;
    if(!chk_(x->right, _sz, _sm)) return false;
    if(_sz != size(x) || fabs(_sm - sum(x)) > eps) {
        return false;
    }
    sz += _sz; sm += _sm;
    return true;
}

bool RBTree::chk() {
    int _sz = 0; 
    double _sm = 0.0; 
    if(!chk_(root, _sz, _sm)) return false;
    if(_sz != size(root) || fabs(_sm - sum(root)) > eps) {
        return false;
    }
    return true;
}

void dfs(node *x) {
    if(x == nullptr) {
        return;
    }
    printf("%f ", x->key);
    dfs(x->left); dfs(x->right);
}

node* RBTree::insert(double val, int idx) {
    node *p, *q;
    node *x = new node(val, idx);
    arr[idx] = x; 
    p = root, q = nullptr;
    if (root == nullptr) {
        x->color = 'b';
        root = x;
    }
    else {
        // x->color = 'r';
        while(p != nullptr) {
            q = p;
            if(p->key < x->key) { //maybe <= ?
                p = p->right;
            }
            else {
                p = p->left;
            }
        }
        x->parent = q;
        if(q->key < x->key)  // maybe <= ?
            q->right = x;
        else    
            q->left = x;
        
        node *temp = x->parent;
        while(temp != nullptr) {
            pull(temp);
            temp = temp->parent;
        }
        insert_fix(x);
    }
    return x;
}

void RBTree::insert_fix(node *x) {
    node *parent, *grantp;
    while((parent = x->parent) != nullptr && parent->color == 'r') {
        grantp = parent->parent;
        if(parent == grantp->left) {
            {
                node *temp = grantp->right;
                if(temp != nullptr && temp->color == 'r') {
                    temp->color = 'b';
                    parent->color = 'b';
                    grantp->color = 'r';
                    x = grantp;
                    continue;
                }
            }
            if(parent->right == x) {
                left_rotate(parent);
                node *temp = parent;
                parent = x;
                x = temp;
            }
            parent->color = 'b';
            grantp->color = 'r';
            right_rotate(grantp);
        }
        else {
            {
                node *temp = grantp->left;
                if(temp != nullptr && temp->color == 'r') {
                    temp->color = 'b';
                    parent->color = 'b';
                    grantp->color = 'r';
                    x = grantp;
                    continue;
                }
            }
            if(parent->left == x) {
                right_rotate(parent);
                node *temp = parent;
                parent = x;
                x = temp;
            }
            parent->color = 'b';
            grantp->color = 'r';
            left_rotate(grantp);
        }
    }
    root->color = 'b';
}

void RBTree::erase(node *p) {
    if(root == nullptr || p == nullptr) {
        return;
    }
    if((p->left != nullptr) && (p->right != nullptr)) {
        node *r = p->right;
        while (r->left != nullptr) 
            r = r->left;
        
        if(p->parent != nullptr) {
            if(p->parent->left == p)
                p->parent->left = r;
            else 
                p->parent->right = r;
        }   
        else {
            root = r;
        }

        node *child = r->right;
        node *parent = r->parent;
        char color = r->color;

        if(parent == p) {
            parent = r;
        }
        else {
            if(child != nullptr) {
                child->parent = parent;
            }
            parent->left = child;
            r->right = p->right;
            p->right->parent = r;
        }
        r->parent = p->parent;
        r->color = p->color;
        r->left = p->left;
        p->left->parent = r;
        
        // pull nodes
        node *temp = parent;
        while(temp != nullptr) {
            pull(temp);
            temp = temp->parent;
        }

        if(color == 'b')
            erase_fix(child, parent);

        delete p;
        return;
    }
    
    node *parent, *child;

    if(p->left != nullptr) 
        child = p->left;
    else 
        child = p->right;

    parent = p->parent;
    char color = p->color;

    if(child != nullptr) {
        child->parent = parent;
    }

    if(parent != nullptr) {
        if(parent->left == p)
            parent->left = child;
        else 
            parent->right = child;
    }
    else {
        root = child;
    }
    
    // pull nodes
    node *temp = parent;
    while(temp != nullptr) {
        pull(temp);
        temp = temp->parent;
    }

    if(color == 'b')
        erase_fix(child, parent);

    delete p;
}

void RBTree::erase(double val) {
    if(root == nullptr) {
        return;
    }
    erase(find(val));
}

void RBTree::erase_fix(node *x, node *p) {
    while((x == nullptr || x->color == 'b') && x != root) {
        if(p->left == x) {
            node *temp = p->right;
            if(temp->color == 'r') {
                temp->color = 'b';
                p->color = 'r';
                left_rotate(p);
                temp = p->right;
            }
            if((temp->left == nullptr || temp->left->color == 'b') &&
               (temp->right == nullptr || temp->right->color == 'b')) {
                temp->color = 'r';
                x = p;
                p = x->parent;
            }
            else {
                if(temp->right == nullptr || temp->right->color == 'b') {
                    temp->left->color = 'b';
                    temp->color = 'r';
                    right_rotate(temp);
                    temp = p->right;
                }
                temp->color = p->color;
                p->color = 'b';
                temp->right->color = 'b';
                left_rotate(p);
                x = root;
                break;
            }
        }
        else {
            node *temp = p->left;
            if(temp->color == 'r') {
                temp->color = 'b';
                p->color = 'r';
                right_rotate(p);
                temp = p->left;
            }
            if((temp->left == nullptr || temp->left->color == 'b') && 
               (temp->right == nullptr || temp->right->color == 'b')) {
                temp->color = 'r';
                x = p;
                p = x->parent;
            }
            else {
                if(temp->left == nullptr || temp->left->color == 'b') {
                    temp->right->color = 'b';
                    temp->color = 'r';
                    left_rotate(temp);
                    temp = p->left;
                }
                temp->color = p->color;
                p->color = 'b';
                temp->left->color = 'b';
                right_rotate(p);
                x = root;
                break;
            }
        }
    }
    if(x != nullptr)
        x->color = 'b';
}

void RBTree::left_rotate(node *x) {
    if(x->right == nullptr)
        return;
    node *y = x->right;
    x->right = y->left;
    if(y->left != nullptr) 
        y->left->parent = x;
    
    y->parent = x->parent;
    if(x->parent != nullptr) {
        if(x == x->parent->left) 
            x->parent->left = y;
        else    
            x->parent->right = y;
    }   
    else {
        root = y;
    }
    y->left = x;
    x->parent = y;
    pull(x); pull(y);
}

void RBTree::right_rotate(node *x) {
    if(x->left == nullptr)
        return;
    node *y = x->left;

    x->left = y->right;
    if(y->right != nullptr)     
        y->right->parent = x;
    
    y->parent = x->parent;

    if(x->parent != nullptr) {    
        if(x == x->parent->left)
            x->parent->left = y;
        else    
            x->parent->right = y;
    }
    else {
        root = y;
    }
    y->right = x;
    x->parent = y;
    pull(x); pull(y);
}

void RBTree::display() {
    printf("Tree:");
    display(root);
    printf("\n");
}

void RBTree::display(node *x) {
    if(x == nullptr) return;
    display(x->left);
    printf(" %f", x->key);
    display(x->right);
}

int RBTree::rank(double val) {
    int res = 0;
    node *p = root;
    while(p != nullptr) {
        if(p->key > val) {
            res += size(p->right) + 1;
            p = p->left;
        }
        else {
            p = p->right;
        }
    }
    return res + 1;
}

int RBTree::rank(node *x) {
    node *y = x; 
    x = x->parent;
    int ret = size(y->right);
    while(x != nullptr) {
        if(x->right != y)
            ret += size(x->right) + 1;
        y = x; x = x->parent;
    }
    return ret + 1;
}

node* RBTree::solve() {
    node *v = root, *w = nullptr;
    int r = size(v->right) + 1;
    double s = sum(v->right) + v->key;
    double c = 1.0 - eps * (size(root) - r);
    while(v != nullptr) {
        if(v->key >= eps * s / c) {
            w = v;
            v = v->left;
            if(v != nullptr) {
                r += size(v->right) + 1;
                s += sum(v->right) + v->key;
                c = 1.0 - eps * (size(root) - r);
            }
        }
        else {
            v = v->right;
            if(v != nullptr) {
                r -= size(v->left) + 1;
                s -= sum(v->left) + v->parent->key;
                c = 1.0 - eps * (size(root) - r);
            }
        }
    }
    return w;
}

double partial_sum(node *x) {
    node *y = x;
    x = x->parent;
    double ret = sum(y->right);
    while(x != nullptr) {
        if(x->right != y)
            ret += sum(x->right) + x->key;
        y = x; x = x->parent;
    }
    return ret;
}

node* RBTree::select_sum(double x) {
    node *p = root;
    while(x >= 0.0) {
        if(sum(p->right) > x) {
            p = p->right;
        }
        else {
            x -= sum(p->right) + p->key;
            if(x < 0.0) return p;
            p = p->left;
        }
    }
    return nullptr;
}

node* RBTree::select_rank(int rk) {
    node *p = root;
    while(p != nullptr) {
        if(size(p->right) >= rk) {
            p = p->right;
        }
        else {
            rk -= size(p->right) + 1;
            if(!rk) return p;
            p = p->left;
        }
    }
    return nullptr;
}

int RBTree::sample() {
    node *w = solve();
    rho = rank(w);
    double lambda = partial_sum(w) / (1.0 - eps * (size(root) - rho));
    double u = dis(gen);
    if(u < 1.0 - eps * (size(root) - rho)) {
        double s = lambda * u;
        node *v = select_sum(s);
        p = v->key / lambda;
        return v->value;
    }
    else {
        u -= 1.0 - eps * (size(root) - rho);
        int r = size(root) - floor(u / eps);
        node *v = select_rank(r);
        p = eps;
        return v->value;
    }
}

double RBTree::getWeight() {
    return 1.0 / (p * n);
}

void RBTree::update(int idx, double val) {
    erase(arr[idx]);
    insert(val, idx);
}