#include "RBTree.h"

extern "C" {


RBTree *T = nullptr;

void getObject(int n, double x) {
    if(T == nullptr) {
        T = new RBTree(n, x);    
    }
    return;
}

void delObject() {
    delete T;
    T = nullptr;
}

void insertObject(double val, int idx) {
    printf("Here\n");
    T->insert(val, idx);
    T->display();
}

void deleteObject(int idx) {

}


}  // extern "C"