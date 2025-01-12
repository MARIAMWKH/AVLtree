# username - Mariamk
# id1      - 212346076
# name1    - Mariam khalaila
# id2      - 209826924
# name2    - Mayar safadi


"""A class represnting a node in an AVL tree"""
import random


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1  # Balance factor
        self.size = 0

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        if self.isRealNode():
            return self.value
        return None

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node):
        self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value

    @type value: str
    @param value: data
    """

    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node

    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        return self.getHeight() != -1

    """gets the size of the node

        @param siz: the size of self, 0 if the node is virtual
        """

    def getSize(self):
        return self.size

    """sets the size of the node

        @type  arg_size: int
        @param  arg_size: the size
        """

    def setSize(self, arg_size):
        self.size = arg_size


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """


    def __init__(self):
        self.size = 0
        self.root = None
        self.Last = None  # the min node
        self.First = None  # the max node
        self.VirtualNode = AVLNode(None)  # the Virtual child
        self.length()

        # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.root is None

    """retrieves the value of the i'th item in the list
    Time complexity: O(log(n))
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

    def retrieve(self, i):
        if i >= self.length() or i < 0:
            return None
        node = self.TreeSelect(i + 1)  # returns the node with the value that's in the index i
        return node.getValue()

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, i, val):
        c = 0
        node = self.createLeaf(val)
        # case 0: empty tree:
        if self.root is None:
            self.root = node
            self.First = node
            self.Last = node
            return 0

        # case 1: if i = self.length()
        if i == self.length():
            last = self.Last
            last.right = node
            node.parent = last
            self.Last = node
        # case 2: i<n
        else:
            if i == 0:
                self.First = node

            temp = self.TreeSelect(i + 1)  # find the current node of rank i+1\
            if not temp.getLeft().isRealNode():
                temp.left = node
                node.parent = temp
            else:  # get Predecessor
                temp = temp.getLeft()
                while temp.getRight().isRealNode():
                    temp = temp.getRight()
                temp.right = node
                node.parent = temp

        y = node.getParent()
        while y is not None:  # fixing the tree
            y.setSize(y.getSize() + 1)
            prev_height = y.getHeight()
            self.updateHeight(y)
            bf = self.calcBF(y)
            if (-2 < bf < 2) and prev_height == y.getHeight():
                y = y.getParent()
                break
            elif -2 < bf < 2:
                c = c + 1
                y = y.getParent()
            else:
                r_numb = self.rotate(y, bf)  # rotates the tree and returns the number of rotations performed
                c = c + r_numb
                if y.getParent is not None:
                    y = y.getParent().getParent()
        while y is not None:  # continue updating the size and height all the way to the root
            y.size += 1
            self.updateHeight(y)
            y = y.getParent()
        self.First = self.TreeSelect(1)
        self.Last = self.TreeSelect(self.length())

        return c

    """deletes the i'th item in the list
    Time Complexity: O(log(n))
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, i):
        rotations = 0
        if i >= self.length() or i < 0:
            return -1
        node = self.TreeSelect(i + 1)  # find the node with the in index i
        rotations = 0

        # the tree have only one node#
        if self.length() == 1:  # the tree have only one node
            self.root = None
            self.First = None
            self.Last = None
            return 0  # no rotation had been occured

        # node is a leaf#
        if (not node.left.isRealNode()) and (not node.right.isRealNode()):  # the node with index i is a leaf
            x = node.parent
            if node.parent.left == node:  # if the node is a left child, delete it
                node.parent.left = self.VirtualNode
            else:  # if the node is a right child, delete it
                node.parent.right = self.VirtualNode

        # the node don't have a right sub tree, only left sub tree#
        elif not node.right.isRealNode():  # node has only left subtree
            if self.root == node:  # if the node is the root and have only left child
                self.root = node.left
                node.left.parent = None
                x = None
            else:
                x = node.parent
                if x.right == node:  # the node is a right child
                    x.right = node.left
                    x.right.parent = x
                else:  # the node is a left child
                    x.left = node.left
                    x.left.parent = x

            node.left = None  # deleting the node
            node.parent = None  # deleting the node

        # the node don't have a left sub tree, only right sub tree#
        elif not node.left.isRealNode():  # node has only right subtree
            if self.root == node:  # if the node is the root and have only right child
                self.root = node.right
                node.right.parent = None
                x = None
            else:
                x = node.parent
                if x.left == node:  # the node is a left child,update
                    x.left = node.right
                    x.left.parent = x
                else:
                    x.right = node.right
                    x.right.parent = x

            node.right = None
            node.parent = None
        # the node has 2 childern#
        else:
            suc_node = self.Successor(node)  # find the successor of node in index i
            if node.right == suc_node:
                if self.root == node:
                    self.root = suc_node
                    suc_node.parent = None
                    suc_node.left = node.left  # connecting with the left subtree
                    if node.left.isRealNode():
                        node.left.parent = suc_node  # connecting with the left subtree
                    x = suc_node
                    self.updateHeight(suc_node)  # updating height
                    self.updateSize(suc_node)  # updating size
                    rotations += 1  # interior rotations

                else:
                    suc_node.left = node.left  # the successor adopting the node left child
                    suc_node.left.parent = suc_node  # the successor adopting the node left child
                    if suc_node.left.isRealNode():
                        suc_node.parent = node.parent  # the parent of node adopt successor
                    if node.parent.right == node:  # if node is a right child
                        node.parent.right = suc_node
                    else:
                        # if node.parent is not None:
                        node.parent.left = suc_node
                    x = suc_node  # node2 that replaced node 1
            else:
                x = suc_node.parent
                if suc_node.right.isRealNode():  # if the successor have a right child ,the right child replaced the successor
                    x.left = suc_node.right
                    x.left.parent = x
                elif x.right == suc_node:  # the successor is a right child, delete it
                    x.right = self.VirtualNode
                else:
                    x.left = self.VirtualNode  # the successor is a left child, delete it
                suc_node.right = node.right  # the successor replacing the node
                suc_node.left = node.left  # the successor replacing the node
                node.right.parent = suc_node  # the successor replacing the node
                node.left.parent = suc_node  # the successor replacing the node
                node.right = None  # deleting the node
                node.left = None  # deleting the node
                if self.root == node:
                    suc_node.parent = None
                    self.root = suc_node
                else:
                    suc_node.parent = node.parent
                    if suc_node.parent.left == node:  # updating parents
                        suc_node.parent.left = suc_node
                    else:
                        suc_node.parent.right = suc_node
                node.parent = None  # node have been deleted
                

        # balancing the tree after the changes and calculating the rotations
        while x is not None:  # the place where we started changes
            prevHeight = x.getHeight()
            self.updateHeight(x)
            balanceF = self.calcBF(x)
            self.updateSize(x)
            if 2 > balanceF > -2:
                if prevHeight == x.getHeight():
                    x = x.parent
                    continue
                else:
                    x = x.parent
                    rotations += 1
                    continue
            else:
                rot_cnt = self.rotate(x, balanceF)
                x = x.parent
                rotations += rot_cnt
        if self.empty():
            self.Last = None
            self.First = None
        else:
            self.First = self.TreeSelect(1)
            self.Last = self.TreeSelect(self.length())
        return rotations

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        if self.empty():
            return None
        return self.First.getValue()

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        if self.empty():
            return None
        return self.Last.getValue()

    """returns an array representing list 
    Time Complexity: O(n)
    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self):
        array = []
        if self.empty():
            return []
        self.ListtoArray_rec(self.root, array)  # calling the recursive method
        return array

    """a recursive helper function that performs an inorder traversal of an AVL tree,and on each node adds the node value to a given list
    Time Complexity: O(n) #n is the number of nodes in the AVL tree. This is because the method performs an inorder traversal of the tree, which visits each node in the tree exactly once.
    @type node:AVL Node
    @type result: Array
    @param result:append node values to result
    """

    def ListtoArray_rec(self, node, result):
        if not node.isRealNode():
            return
        if node is not None:
            self.ListtoArray_rec(node.getLeft(), result)
            result.append(node.value)
            self.ListtoArray_rec(node.getRight(), result)

    """returns the size of the list 
    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        if self.empty():
            return 0
        return self.root.getSize()

    """sort the info values of the list
    Time complexity: o(nlog(n))
    @rtype: list
    @type value_arr: Array
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    """

    def sort(self):
        values_arr = self.listToArray()  # get the values of the AVLTree list in ordered array
        lst = self.merge_sort(values_arr)  # sorting by values using quicksort method
        new_sorted_tree = AVLTreeList()  # new AVLTreeList we will use it to return the sorted avltreelist
        for i in range(0, len(values_arr)):
            new_sorted_tree.insert(i, lst[i])
        return new_sorted_tree

    """permute the info values of the list 
    Time Complexity: O(nlog(n))
    @:param values:array that contain the values of the self nodes 
    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    """

    def permutation(self):
        values = self.listToArray()  # get the values in sorted order
        lst = self.array_shuffle(values)  # Shuffle the values
        new_shuffled_tree = AVLTreeList()  # create a new AVLtree
        for i in range(0, len(values)):  # Build a new AVL tree from the shuffled values and indexes
            new_shuffled_tree.insert(i, lst[i])
        return new_shuffled_tree

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        if lst.empty():
            if self.empty():
                return 0
            else:
                return self.root.getHeight()

        if self.empty():
            self.root = lst.root
            self.First = lst.First
            self.Last = lst.Last
            return lst.root.getHeight()

        height1 = self.root.getHeight()
        height2 = lst.root.getHeight()
        dh = abs(self.root.getHeight() - lst.root.getHeight())
        if self.length() == 1:
            lst.insert(0, self.root.getValue())
            self.First = lst.First
            self.root = lst.root
            self.Last = lst.Last
            return dh
        x = self.Last
        self.delete(self.length() - 1)
        self.Last = lst.Last
        self.root = self.join(self.getRoot(), x, lst.getRoot())
        return dh

    """joins the subtree t1 to the left branch of t2 though x ,return the root of the joined subtree
        Time Complexity: O(log(n)) - n the number of nodes in t2
        @type t1: AVLNode
        @param t1: root of the first sub tree
        @type t2: AVLNode
        @param t2: root of the second sub tree
        @type x: AVLNode
        @param x: node to connect t1 and t2
        @rtype: AVLNode
        @returns: root of the joined subtree
        """
    def join(self, t1, x, t2):
        if t1 is None:
            t1 = self.VirtualNode
        if t2 is None:
            t2 = self.VirtualNode
        height1 = t1.getHeight()
        height2 = t2.getHeight()
        if height1 < height2:
            temp = t2
            while height1 < temp.getHeight():
                temp = temp.getLeft()
            x.setRight(temp)
            x.setLeft(t1)
            x.setParent(temp.getParent())
            if temp.getParent() is not None:
                temp.getParent().setLeft(x)
            temp.setParent(x)
            if t1.isRealNode():
                t1.setParent(x)
        else:
            temp = t1
            while height2 < temp.getHeight():
                temp = temp.getRight()
            x.setLeft(temp)
            x.setRight(t2)
            x.setParent(temp.getParent())
            if temp.getParent() is not None:
                temp.getParent().setRight(x)
            temp.setParent(x)
            if t2.isRealNode():
                t2.setParent(x)
        temp_parent = x
        self.balance_after_join(temp_parent)
        while temp_parent.parent is not None:
            temp_parent = temp_parent.parent
        return temp_parent

    """balances the joined tree from the inserted node 
        Time Complexity: O(log(n))
        @type node: AVLNode
        @param node: node to to start balancing from upwards
        """

    def balance_after_join(self, node):
        while node is not None:
            prev = node.getHeight()
            self.updateHeight(node)
            node.size = node.left.size + node.right.size + 1
            bf = self.calcBF(node)
            if (2 > bf > -2) and prev == node.getHeight():
                node = node.parent
                break
            elif 2 > bf > -2:
                node = node.parent
            else:
                self.rotate(node, bf)
                if node.parent is None:
                    break
                if node.parent is not None:
                    node = node.parent.parent
        while node is not None:  # continue updating the size and height all the way to the root
            node.size += 1
            self.updateHeight(node)
            node = node.getParent()

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        global index
        global count
        count = 0
        index = -1
        if self.empty():
            return -1
        self.search_rec(self.root, val)
        return index

    """recursive searches for a *value* in the list

        @type val: str
        @param val: a value to be searched
        @rtype: None
        @returns: None, stops when we fine the needed val
        """

    def search_rec(self, node, val):
        global index
        global count
        if node.isRealNode():
            self.search_rec(node.getLeft(), val)
            if node.value == val:
                index = count
                return
            count += 1
            self.search_rec(node.getRight(), val)
        else:
            return

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """ helping functions"""

    """rotates node B to the right
            Time Complexity: O(1)
            @type B: AVLNode
            @param B: the node to be rotated
        """

    def right_rotate(self, B):
        A = B.getLeft()
        B.setLeft(A.getRight())
        if B.getLeft() is not None:
            B.getLeft().setParent(B)
        A.setRight(B)
        A.setParent(B.getParent())
        if B.getParent() is None:  # edge case: rotating the root or rotating the root of a subtree
            self.root = A

        elif A.getParent().getLeft() == B:
            A.getParent().setLeft(A)
        else:
            A.getParent().setRight(A)
        B.setParent(A)
        # update sizes of the nodes affected by the rotation
        A.setSize(B.getSize())
        B.setSize(B.getLeft().getSize() + B.getRight().getSize() + 1)
        # update heights on the nodes affected by the rotation
        B.setHeight(1 + max(B.getLeft().getHeight(), B.getRight().getHeight()))
        A.setHeight(1 + max(A.getLeft().getHeight(), A.getRight().getHeight()))

        """rotates node to the left
            Time Complexity: O(1)
            @type node: AVLNode
            @param node: the node to be rotated
        """

    def left_rotate(self, B):
        A = B.getRight()
        B.setRight(A.getLeft())
        if B.getRight().isRealNode():
            B.getRight().setParent(B)
        A.setLeft(B)
        A.setParent(B.getParent())
        if B.getParent() is None:  # edge case: rotating the root or rotating the root of a subtree
            self.root = A

        elif A.getParent().getLeft() == B:
            A.getParent().setLeft(A)
        else:
            A.getParent().setRight(A)
        B.setParent(A)
        # update sizes of the nodes affected by the rotation
        A.setSize(B.getSize())
        B.setSize(B.getLeft().getSize() + B.getRight().getSize() + 1)
        # update heights on the nodes affected by the rotation
        B.setHeight(1 + max(B.getLeft().getHeight(), B.getRight().getHeight()))
        A.setHeight(1 + max(A.getLeft().getHeight(), A.getRight().getHeight()))

    """rotates node if necessary acccording to the balance factor given and returns 1 if one rotation was performed 
    and 2 if two rotations were performed 
    Time Complexity: O(1)
     @type node: AVLNode
     @param node: node for which the height is updated
     @type bf: int
    @param bf: balance factor of node
    @rtype: int
     @returns: 1 if one rotation was performed and 2 if two rotations were performed
    """

    def rotate(self, node, bf):
        c = 0
        if bf == -2:
            Rbf = self.calcBF(node.getRight())
            if Rbf == -1:
                self.left_rotate(node)
                c = 1
            elif Rbf == +1:
                self.right_rotate(node.getRight())
                self.left_rotate(node)
                c = 2

        else:
            Lbf = self.calcBF(node.getLeft())
            if Lbf == -1:
                self.left_rotate(node.getLeft())
                self.right_rotate(node)
                c = 2

            else:
                self.right_rotate(node)
                c = 1
        return c

    """returns the node in index rank
        Time Complexity: O(log(n))
        @type rank: int
        @param rank: the rank of the node to be returned
        @rtype: AVLNode
        @returns: the rank'th smallest node in the tree
    """

    def TreeSelect(self, rank):
        return self.TreeSelectRec(self.root, rank)

    """recursive method that help us to implement TreeSelect method 
            Time Complexity: O(log(n))
            @type node: AVLNode
            @type rank: int
            @param rank: the rank of the node to be returned
            @rtype: AVLNode
            @returns: the rank'th smallest node in the tree
        """

    def TreeSelectRec(self, node, rank):
        temp = node.getLeft().getSize()
        if temp + 1 == rank:
            return node
        if temp >= rank:
            return self.TreeSelectRec(node.getLeft(), rank)
        return self.TreeSelectRec(node.getRight(), rank - temp - 1)

    """returns the balance factor of a node
            Time Complexity: O(1)
            @type node: AVLNode
            @param node: node for which the function calculate the balance factor
            @rtype: int
            @returns: the branch factor of node
        """

    def calcBF(self, node):
        return node.left.getHeight() - node.right.getHeight()
         

    """creates a new node with given value,with two virtual sons 
                Time Complexity: O(1)
                @type val: str
                @param val: value of the new node 
                @returns: new node with given value,with two virtual sons
            """

    def createLeaf(self, val):
        leaf = AVLNode(val)  # create node with given value
        leaf.left = AVLNode(None)
        leaf.right = AVLNode(None)
        leaf.height = 0
        leaf.size = 1
        return leaf

    """updates the Height of a node
            Time Complexity: O(1)
            @type node: AVLNode
            @param node: node that we have to update his height
        """

    def updateHeight(self, node):
        node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)

    """shuffling an array using random 
    Time Complexity: o(n)
    @type arr: Array
    @return: the new shuffled array
    """

    def array_shuffle(self, arr):
        n = len(arr)
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    """updates the Size of a node
                    Time Complexity: O(1)
                    @type node: AVLNode
                    @param node: node that we have to update his Size
                """

    def updateSize(self, node):
        node.setSize(node.left.getSize() + node.right.getSize() + 1)

    """sorting an array  
        Time Complexity: o(nlogn)
        @type self:AVLTreeList
        @type arr: Array
        @return: the new sorted array
        """

    def merge_sort(self, arr):
        # base case: if the length of the array is 1 or 0, return the array
        if len(arr) <= 1:
            return arr

        # split the array into two halves
        mid = len(arr) // 2
        left_one = arr[:mid]
        right_one = arr[mid:]

        # recursively sort the two halves
        left_one = self.merge_sort(left_one)
        right_one = self.merge_sort(right_one)

        # merge the sorted halves and return the result
        return self.merge(left_one, right_one)

    def merge(self, left_one, right_one):
        # create an empty list to store the merged list
        merged = []

        # iterate over the lists until one of them is empty
        while left_one and right_one:
            # compare the first elements of the lists and append the smaller one
            # to the merged list
            if left_one[0] < right_one[0]:
                merged.append(left_one.pop(0))
            else:
                merged.append(right_one.pop(0))
        # append any remaining elements from the non-empty list
        merged.extend(left_one)
        merged.extend(right_one)
        return merged


    """get the height of tree self 
    Time complexity:o(1)
    @type self:AVLTreeList
    @return: the height of the tree
    """

    def getTreeHeight(self):
        return self.getRoot().getHeight()

    """returns the successor of the node in index i
            Time Complexity: O(log(n))
            @type i: int
            @param i: index of node to which the function finds the successor
            @rtype: AVLNode
            @returns: successor of node in index i
        """
    def Successor(self, node):
        if node.right.isRealNode():
            return self.minValue(node.right)
        mom = node.parent
        while mom is not None and node == mom.right:
            node = mom
            mom = node.parent
        return mom

    """ recursive method that find the min node in given treeTime Complexity: O(log(n))
                @type node: AVLNode
                @rtype: AVLNode
                @return: minimal node

    """

    def minValue(self, node):
        current = node
        # loop down to find the leftmost leaf
        while current.isRealNode():
            if not current.left.isRealNode():
                break
            current = current.left
        return current
