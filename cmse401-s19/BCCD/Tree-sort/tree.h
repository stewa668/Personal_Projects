/* Serial and parallel binary tree traversal
 * Part of Parallelization: Binary Tree Traversal, Blue Waters Undergraduate 
 *  Petascale Module
 *
 * Aaron Weeden and Patrick Royal, Shodor Education Foundation, Inc.
 * April 2012
 */

/* Define node and stack structures */
struct node {
    int label;
    struct node * left;
    struct node * right;
};

struct stack {
    struct node ** nodes;
    int size;
};

/* Functions */

/* Build a tree */
void buildtree(struct node ** root, int root_label, int row_num, int height) {
    if(row_num < height) {
        /* Allocate memory for the root and set its label */
        *root = malloc(sizeof(struct node));
        (*root)->label = root_label;

        buildtree(&((*root)->left),  root_label - pow(2, height-row_num-2), 
                row_num + 1, height);
        buildtree(&((*root)->right), root_label + pow(2, height-row_num-2),
                row_num + 1, height);
    } else {
        *root = NULL;
    }
    return;
}

/* Free a tree's memory */
void freetree(struct node ** root) {
    if((*root) != NULL) {
        freetree(&((*root)->left));
        freetree(&((*root)->right));
        free((*root));
    }
    return;
}


/* Push an element to the top of the stack */
void push(struct stack * s, struct node * n) {
    s->nodes[s->size++] = n;
}

/* Pop an element from the top of the stack and return it */
struct node * pop(struct stack * s) {
    if(s->size == 0) return NULL;
    else return s->nodes[--(s->size)];
}

/* Return the first element from the top of the stack without popping */
struct node * peek(struct stack s) {
    if(s.size == 0) return NULL;
    else return s.nodes[s.size-1];
}
