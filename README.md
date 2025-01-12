# AVLtree
Python implementation of a List ADT using AVL Tree, supporting efficient O(log n) operations including insert, delete, retrieve, concat and sort. Built for Data Structures course.
# AVL Tree List Implementation

A Python implementation of a List ADT (Abstract Data Type) using a self-balancing AVL Tree. This implementation ensures O(log n) time complexity for major operations.

## Features

- Self-balancing AVL Tree structure
- Efficient O(log n) operations for:
  - Insertion
  - Deletion
  - Retrieval
  - Concatenation
- Additional functionality:
  - Sorting
  - Random permutation
  - List to array conversion

## Implementation Details

The implementation consists of two main classes:
- `AVLNode`: Represents a node in the AVL tree with balancing information
- `AVLTreeList`: Implements the List ADT using an AVL tree structure

### Key Methods

- `insert(i, val)`: Insert value at index i
- `delete(i)`: Delete item at index i
- `retrieve(i)`: Get value at index i
- `concat(lst)`: Concatenate another list
- `sort()`: Sort the list values
- `permutation()`: Create random permutation of list

## Time Complexities

- Insert: O(log n)
- Delete: O(log n)
- Retrieve: O(log n)
- Concat: O(log n)
- Sort: O(n log n)
- Search: O(n)

## Usage Example

```python
# Create a new AVL Tree List
avl_list = AVLTreeList()

# Insert elements
avl_list.insert(0, "first")
avl_list.insert(1, "second")
avl_list.insert(2, "third")

# Retrieve element
value = avl_list.retrieve(1)  # returns "second"

# Delete element
avl_list.delete(0)  # removes "first"
```

## Contributors

- Mariam khalaila 
- Mayar safadi
