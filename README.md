# LeetCode NoteBook

**<u>继Solution76之后的笔记，其余详见notability</u>**

**目前开始跟进刷残酷刷题群的每日打卡题**

https://docs.google.com/spreadsheets/d/1kBGyRsSdbGDu7DzjQcC-UkZjZERdrP8-_QyVGXHSrB8/edit#gid=0

# Table Of Content

1. <a href="#Tree-Content">Tree</a>
2. <a href="#Backtracking-Content">Backtracking</a>

# <a id="Tree-Content">Tree</a>

## [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

> Using BFS or DFS, search the whole tree by level. According to the BST, each node's left node is smaller than it, right node is greater than it. So using BFS/DFS search, the first node with a value between nodeA's value and nodeB's value is the lowest common ancestor.

做过[二叉树：公共祖先问题](https://programmercarl.com/0236.二叉树的最近公共祖先.html)题目的同学应该知道，利用回溯从底向上搜索，遇到一个节点的左子树里有p，右子树里有q，那么当前节点就是最近公共祖先。

那么本题是二叉搜索树，二叉搜索树是有序的，那得好好利用一下这个特点。

在有序树里，如果判断一个节点的左子树里有p，右子树里有q呢？

其实只要从上到下遍历的时候，cur节点是数值在[p, q]区间中则说明该节点cur就是最近公共祖先了。

理解这一点，本题就很好解了。

和[二叉树：公共祖先问题](https://programmercarl.com/0236.二叉树的最近公共祖先.html)不同，普通二叉树求最近公共祖先需要使用回溯，从底向上来查找，二叉搜索树就不用了，因为搜索树有序（相当于自带方向），那么只要从上向下遍历就可以了。

```java
//直接用前序遍历，找到第一个满足val>=p.val&&val<=q.val的就是公共祖先
    TreeNode ancestor = new TreeNode();
    boolean isFind = false;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(q.val<p.val){
            TreeNode temp = p;
            p=q;
            q=temp;
        }
        checkAncestor(root, p, q);
        return ancestor;
    }
    private void checkAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null){
            return;
        }
        if(!isFind){
            if(root.val>=p.val&&root.val<=q.val){
                ancestor = root;
                isFind=true;
            }
        }
        checkAncestor(root.left, p, q);
        checkAncestor(root.right, p, q);
    }
//递归2
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        if (root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
        return root;
    }
```

如果root的val大于p和q的val，那说明公共节点应该往左子树去，vice versa。



## [0701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> From root node, if insert node's value greater than current nodes' value, then search the right node to do the recursion, and vice versa. Accoding to this rule, when find a empty node, then the insert node can just be inserted to this position.

递归三部曲：

- 确定递归函数参数以及返回值

参数就是根节点指针，以及要插入元素，这里递归函数要不要有返回值呢？

可以有，也可以没有，但递归函数如果没有返回值的话，实现是比较麻烦的，下面也会给出其具体实现代码。

**有返回值的话，可以利用返回值完成新加入的节点与其父节点的赋值操作**。（下面会进一步解释）

递归函数的返回类型为节点类型TreeNode * 。

代码如下：

```java
TreeNode* insertIntoBST(TreeNode* root, int val)
```

- 确定终止条件

终止条件就是找到遍历的节点为null的时候，就是要插入节点的位置了，并把插入的节点返回。

代码如下：

```java
if (root == NULL) {
    TreeNode* node = new TreeNode(val);
    return node;
}
```

这里把添加的节点返回给上一层，就完成了父子节点的赋值操作了，详细再往下看。

- 确定单层递归的逻辑

此时要明确，需要遍历整棵树么？

别忘了这是搜索树，遍历整颗搜索树简直是对搜索树的侮辱，哈哈。

搜索树是有方向了，可以根据插入元素的数值，决定递归方向。

代码如下：

```java
if (root->val > val) root->left = insertIntoBST(root->left, val);
if (root->val < val) root->right = insertIntoBST(root->right, val);
return root;
```

**到这里，大家应该能感受到，如何通过递归函数返回值完成了新加入节点的父子关系赋值操作了，下一层将加入节点返回，本层用root->left或者root->right将其接住**。

```java
public TreeNode insertIntoBST(TreeNode root, int val) {
        return checkInsertBST(root, val);
    }
    private TreeNode checkInsertBST(TreeNode root, int val) {
        if(root==null){
            return new TreeNode(val);
        }
        if(root.val>val){
            root.left=checkInsertBST(root.left, val);
        }
        if(root.val<val){
            root.right=checkInsertBST(root.right, val);
        }
        return root;
    }
```







## [0450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

- 确定终止条件

遇到空返回，其实这也说明没找到删除的节点，遍历到空节点直接返回了

```java
if (root == null){
		return root;
} 
```

- 确定单层递归的逻辑

这里就把平衡二叉树中删除节点遇到的情况都搞清楚。

有以下五种情况：

- 第一种情况：没找到删除的节点，遍历到空节点直接返回了
- 找到删除的节点
  - 第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
  - 第三种情况：删除节点的左孩子为空，右孩子不为空，删除节点，右孩子补位，返回右孩子为根节点
  - 第四种情况：删除节点的右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
  - 第五种情况：左右孩子节点都不为空，则将删除节点的左子树头结点（左孩子）放到删除节点的右子树的最左面节点的左孩子上，返回删除节点右孩子为新的根节点(看下面代码部分)

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f747661312e73696e61696d672e636e2f6c617267652f30303865476d5a456c7931676e626a336b3539366d673330647130616967797a2e676966.png" alt="450.删除二叉搜索树中的节点"  />

```java
//0450. 删除二叉搜索树中的节点
public class Solution79 {
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root==null){
            return root;
        }
        if(root.val>key){
            root.left=deleteNode(root.left, key);
        }else if(root.val<key){
            root.right=deleteNode(root.right, key);
        }else{
            if(root.left==null&&root.right==null){
                return null;
            }else if(root.left==null){
                return root.right;
            }else if(root.right==null){
                return root.left;
            }else{
                findLastLeft(root.right).left=root.left;
                return root.right;
            }
        }
        return root;
    }
    //这部分就是用来找到被删除节点右子树最左面节点的
    private TreeNode findLastLeft(TreeNode root){
        if(root.left==null){
            return root;
        }
        return findLastLeft(root.left);
    }
}
```

## [0669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

```java
//0669. 修剪二叉搜索树
public class Solution80 {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if(root==null){
            return root;
        }
        if(root.val>=low&&root.val<=high){
            root.left=trimBST(root.left, low, high);
            root.right=trimBST(root.right, low, high);
        }else {
            if(root.val>high){
                return trimBST(root.left, low, high);
            }else if(root.val<low){
                return trimBST(root.right, low, high);
            }
        }
        return root;
    }
}
```

思路：

题目首先给定了low和high，在val值在[low, high]中的节点都会被保留下来，而其余的就要被删除

- 根据二叉搜索树的特性，val<low的节点，意味着它的左子树上所有节点也一样要被抹除，因为左子树上的所有节点均小于根节点。
- val>high的节点，意味着它的右子树上所有节点都要被抹除，因为右子树上的所有节点值均大于根节点。



## [0108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

```java
//0108. 将有序数组转换为二叉搜索树
public class Solution81 {
    public TreeNode sortedArrayToBST(int[] nums) {
        return checkToBST(nums, 0, nums.length-1);
    }

    private TreeNode checkToBST(int[] nums, int left, int right) {
        if(left>right){
            return null;
        }
        if(left==right){
            return new TreeNode(nums[(left+right)/2]);
        }
        int mid = (left+right)/2;
        TreeNode curNode = new TreeNode(nums[mid]);
        curNode.left = checkToBST(nums, left, mid-1);
        curNode.right = checkToBST(nums, mid+1, right);
        return curNode;
    }
}
```

思路：

递归中传入数据分别有nums，以及记录当前节点段的left和right双指针。

因为题目要求是创建高度平衡二叉搜索树，而数组也是有序数组，实际上这题目可以等价于创建高度平衡二叉树，数组是一个任意的数组，因为为了高度平衡，那么就是每次取数组中间的节点当作root，以此来递归，代码如下：

```java
				int mid = (left+right)/2;
        TreeNode curNode = new TreeNode(nums[mid]);
        curNode.left = checkToBST(nums, left, mid-1);
        curNode.right = checkToBST(nums, mid+1, right);
```

终止条件就是(left>right or left==right)两种情况时。

```java
				if(left>right){
            return null;
        }
        if(left==right){
            return new TreeNode(nums[(left+right)/2]);
        }
```



## [0538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

```java
//0538. 把二叉搜索树转换为累加树
public class Solution82 {
    private int sum = 0;
    public TreeNode convertBST(TreeNode root) {
        checkBST(root);
        return root;
    }
    private void checkBST(TreeNode root) {
        if(root==null){
            return;
        }
        checkBST(root.right);
        root.val+=sum;
        sum=root.val;
        checkBST(root.left);
        return;
    }
}
```

思路：

根据二叉搜索树的特性，按照中序遍历它是一个有序的递增数组，那么如果将其按照右中左的顺序来遍历，得到的就是一个有序的递减数组，只需要将从第一位开始的元素累加即可得到这个累加树，代码如上，非常简单。







## [0103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

```java
//0103. 二叉树的锯齿形层序遍历
public class Solution1 {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.addLast(root);
        Boolean isClockwise = true;
        while (!deque.isEmpty()) {
            List<Integer> innerList = new ArrayList<>();
            int size = deque.size();
            if (isClockwise) {
                for (int i = 0; i < size; i++) {
                    TreeNode cur = deque.pollFirst();
                    innerList.add(cur.val);
                    if (cur.left != null) {
                        deque.addLast(cur.left);
                    }
                    if (cur.right != null) {
                        deque.addLast(cur.right);
                    }
                }
                isClockwise = false;
                res.add(innerList);
                continue;
            }else {
                for (int i = 0; i < size; i++) {
                    TreeNode cur = deque.pollLast();
                    innerList.add(cur.val);
                    if (cur.right != null) {
                        deque.addFirst(cur.right);
                    }
                    if (cur.left != null) {
                        deque.addFirst(cur.left);
                    }
                }
                isClockwise = true;
                res.add(innerList);
                continue;
            }
        }
        return res;
    }
}
```













## [0199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

```java
	private void checkList(TreeNode node) {
        if (node == null) {
            return;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.addLast(node);
        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 1; i <= deque.size(); i++) {
                TreeNode temp = deque.pollFirst();
                if (i == deque.size()) {
                    res.add(temp.val);
                }
                if (temp.left != null) {
                    deque.addLast(temp.left);
                }
                if (temp.right != null) {
                    deque.addLast(temp.right);
                }
            }
        }
    }
```

思路：

广度优先算法。就是层序遍历树，这样从左往右遍历每一层，那最后的那个元素就是右视图中的一部分。



```java
	List<Integer> res = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {
        Set<Integer> set = new HashSet<>();
        //checkList(root);
        checkList2(root, 0, set);
        return res;
    }
	//深度优先算法
    private void checkList2(TreeNode node, int depth, Set set) {
        if (node == null) {
            return;
        }
        if (!set.contains(depth)) {
            set.add(depth);
            res.add(node.val);
        }
        checkList2(node.right, depth + 1, set);
        checkList2(node.left, depth + 1, set);
    }
```

思路：

深度优先算法。这可以写成一个递归，一直先向右节点遍历，同时还要记录一个深度，利用set来判断新的深度的出现，当满足`!set.contains(depth)`时，说明出现了新的一层，同时由于又是先往右遍历，所以当前节点一定是最右节点，也就是右视图的一部分。











## [0124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```java
//0124. 二叉树中的最大路径和
public class Solution2 {
    int max = -1001;
    public int maxPathSum(TreeNode root) {
        checkMaxPath(root);
        return max;
    }

    private int checkMaxPath(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = checkMaxPath(root.left);
        int right = checkMaxPath(root.right);
        max = max > root.val + left + right ? max : root.val + left + right;
        int maxReturn = left + root.val > right + root.val ? left + root.val : right + root.val;
        if (maxReturn < 0) {
            maxReturn = 0;
        }
        return maxReturn;
    }
}
```

思路：

走左中右的遍历顺序去遍历整个树：

1. 当遍历到root == null时return 0

2. 记录左子树的最大返回值和右子树的最大返回值

   ```java
   		int left = checkMaxPath(root.left);
   		int right = checkMaxPath(root.right);
   		int maxReturn = left + root.val > right + root.val ? left + root.val : right + root.val;
           if (maxReturn < 0) {
               maxReturn = 0;
           }
   ```

   路径就是左边和右边只能选一条路走，所以取更大的那个为返回值，同时如果返回值小于0，则直接放弃这条路不走，所以会有下面的那个`if (maxReturn < 0) maxReturn = 0`

3. 最后用`max = max > root.val + left + right ? max : root.val + left + right`用来更新最大路径和





## [0129. 求根节点到叶节点数字之和](https://leetcode.cn/problems/sum-root-to-leaf-numbers/)

```java
	List<String> res = new ArrayList<>();
    public int sumNumbers(TreeNode root) {
        checkSum(root, new String());
        int sum = 0;
        for (String num : res) {
            sum += Integer.parseInt(num);
        }
        return sum;
    }

    private void checkSum(TreeNode node, String s) {
        StringBuilder sb = new StringBuilder(s);
        sb.append(Integer.toString(node.val));
        if (node.left == null && node.right == null) {
            res.add(sb.toString());
            return;
        }
        if (node.left != null) {
            checkSum(node.left, sb.toString());
        }
        if (node.right != null) {
            checkSum(node.right, sb.toString());
        }
    }
```

两种做法，深度优先和广度优先

1. 深度优先的做法就是用递归
2. 广度优先的做法就是用层序遍历



## [0105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```java
public TreeNode buildTree(int[] preorder, int[] inorder) {
        return checkTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }
    private TreeNode checkTree(int[] preorder, int preLeft, int preRight, int[] inorder, int inLeft, int inRight) {
        //防止越界
        if(preRight-preLeft<1){
            return null;
        }
        if (preRight - preLeft == 0) {
            return new TreeNode(preorder[preLeft]);
        }
        int count = 0;
        while (preorder[preLeft] != inorder[inLeft + count]) {
            count++;
        }
        TreeNode curNode = new TreeNode(preorder[preLeft]);
        curNode.left = checkTree(preorder, preLeft + 1, preLeft + count, inorder, inLeft, inLeft + count - 1);
        curNode.right = checkTree(preorder, preLeft + count + 1, preRight, inorder, inLeft + count + 1, inRight);
        return curNode;
    }
```

**递归法：**

对于任意一颗树而言，前序遍历的形式总是：

> [ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]

即根节点总是前序遍历中的第一个节点。而中序遍历的形式总是

> [ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]

只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。

这样以来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，我们就可以递归地对构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。

**细节**

在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。我们可以考虑使用哈希表来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 O(1)的时间对根节点进行定位了。



## [0101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

```java
	public boolean isSymmetric(TreeNode root) {
        return checkSym(root, root);
    }

    private Boolean checkSym(TreeNode p1, TreeNode p2) {
        if (p1 == null && p2 == null) {
            return true;
        }
        if (p1 == null || p2 == null) {
            return false;
        }
        return p1.val == p2.val && checkSym(p1.left, p2.right) && checkSym(p1.right, p2.left);
    }
```

递归和迭代都行

对称实际上就是左右子树镜像，那就去遍历整个二叉树，左子树往左遍历，右子树就往右遍历，相反，左子树往右遍历，右子树就往左遍历。









## [0096. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

```java
public int numTrees(int n) {
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i < dp.length; i++) {
        for (int j = 1; j <= i; j++) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    return dp[n];
}
```

动态规划

- 假设n个节点存在的二叉搜索树的个数是dp[n]，令f(i)为以i为根节点所能组成的二叉搜索树的个数，则：

  `G(n) =  f(1) + f(2) + f(3) + f(4) + f(5) + ... + f(n)`

- 而当i是根节点时，其左子树含有i-1个节点，右子树含有n-i哥节点，所以其f(i)---以i为根节点所能组成的二叉搜索树的个数可以等价为求组合数：

  `f(i) = G(i - 1) * G(n - i)`

- 以此就找到了动态递归的公式







## [0114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

```java
public void flatten(TreeNode root) {
    if (root == null) {
        return;
    }
    checkFlatten(root);
}

private void checkFlatten(TreeNode root) {
    if (root.left == null && root.right == null) {
        return;
    }
    if (root.left != null) {
        checkFlatten(root.left);
    }
    TreeNode temp = root.right;
    root.right = root.left;
    root.left = null;
    while (root.right != null) {
        root = root.right;
    }
    root.right = temp;
    if (root.right != null) {
        checkFlatten(root.right);
    }
}
```

因为题目要求使用原地算法（`O(1)` 额外空间）展开这棵树

其实可以把整个问题分解成非常多个小问题：

- root的左子树提前排列成单链表，然后把排序好的左子树插入到root与右子树之间

  ```java
  TreeNode temp = root.right;
  root.right = root.left;
  root.left = null;
  while (root.right != null) {
      root = root.right;
  }
  root.right = temp;
  ```

  





## [0208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

```java
private Trie[] childrens;

private Boolean isEnd;

public Trie() {
    this.childrens = new Trie[26];
    this.isEnd = false;
}

public void insert(String word) {
    Trie node = this;
    for (int i = 0; i < word.length(); i++) {
        Character c = word.charAt(i);
        int index = c - 'a';
        if (node.childrens[index] == null) {
            node.childrens[index] = new Trie();
        }
        node = node.childrens[index];
    }
    node.isEnd = true;
}

public boolean search(String word) {
    Trie node = this;
    for (int i = 0; i < word.length(); i++) {
        Character c = word.charAt(i);
        int index = c - 'a';
        if (node.childrens[index] != null) {
            node = node.childrens[index];
        } else {
            return false;
        }
    }
    return node.isEnd;
}

public boolean startsWith(String prefix) {
    Trie node = this;
    for (int i = 0; i < prefix.length(); i++) {
        Character c = prefix.charAt(i);
        int index = c - 'a';
        if (node.childrens[index] != null) {
            node = node.childrens[index];
        } else {
            return false;
        }
    }
    return true;
}
```

前缀树又称字典树

其节点数据结构应该是这样的：

```java
public class Trie() {
    private Trie[] childrens;
    
    private Boolean isEnd;
}
```

`isEnd`表示该节点是否是一个串的结束，`childrens`代表的是26个字母，每一个节点的下面有26个分叉，只不过很多分叉是空值。

**插入功能：**

- 遍历插入的字符串
  - 如果遍历位置的字母在当前节点的childrens子节点中是null，说明之前还没有该前缀存在，因此插入进去，并将当前cur节点位置更新`node = node.childrens[index];`，在循环结束时，当前cur节点就是字符串的末尾，所以把这个node的isEnd属性设置为true。

**查询完整字符串功能：**

- 遍历待匹配字符串
  - 从根节点的子节点开始，一直向下匹配，如果出现节点值为空，则返回false，如果循环结束，即所有字符都匹配，这时还要判断该节点的isEnd是否为true，根据isEnd返回结果。

**查询prefix功能：**

该功能其实和上面的基本一样，只不过不需要判断节点的isEnd。









## [0655. 输出二叉树](https://leetcode.cn/problems/print-binary-tree/)

```java
public int depth;
public List<List<String>> printTree(TreeNode root) {
    if (root == null) {
        return new ArrayList<>();
    }
    depth = dfsForDepth(root, 1);
    int col = (int) Math.pow(2, depth) - 1;
    List<List<String>> list = new ArrayList<>();
    for (int i = 0; i < depth; i++) {
        List<String> curList = new ArrayList<>();
        for (int j = 0; j < col; j++) {
            curList.add("");
        }
        list.add(curList);
    }
    list = dfsConstruct(root, 1, list, (col - 1) / 2);
    return list;
}
//构造满足题目要求的二维矩阵
private List<List<String>> dfsConstruct(TreeNode root, int curDepth, List<List<String>> list, int mid) {
    int pow = (int) Math.pow(2, depth - curDepth - 1);
    List<String> curList = list.get(curDepth - 1);
    curList.set(mid, Integer.toString(root.val));
    list.set(curDepth - 1, curList);
    if (root.left == null && root.right == null) {
        return list;
    }
    if (root.left != null) {
        list = dfsConstruct(root.left, curDepth + 1, list, mid - pow);
    }
    if (root.right != null) {
        list = dfsConstruct(root.right, curDepth + 1, list, mid + pow);
    }
    return list;
}
//求得树的深度
private Integer dfsForDepth(TreeNode root, int depth) {
    if (root.left == null && root.right == null) {
        return depth;
    }
    int depthLeft = 0;
    int depthRight = 0;
    if (root.left != null) {
        depthLeft = dfsForDepth(root.left, depth + 1);
    }
    if (root.right != null) {
        depthRight = dfsForDepth(root.right, depth + 1);
    }
    return Math.max(depthLeft, depthRight);
}
```

这题首先需要去得出树的深度，利用一次dfs或者bfs都可以得出深度

在知道深度之后根据深度：

1. 求得每一个节点所对应的二维矩阵中的位置

   1. 第一维度就是当前节点的深度

   2. 第二维度可以根据父节点的位置来确定（若父节点为`matrix[r][c]`）

      1. 左节点： `res[r+1][c-2^(height-r-1)]`
      2. 右节点：`res[r+1][c+2^(height-r-1)]`

      









## [0297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

```java
public class Solution9 {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return doSerialize(root, "");
    }

    private String doSerialize(TreeNode root, String cur) {
        if (root == null) {
            return cur + "None,";
        }
        cur += root.val + ",";
        cur = doSerialize(root.left, cur);
        cur = doSerialize(root.right, cur);
        return cur;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] strings = data.split(",");
        List<String> stringList = new LinkedList<>(Arrays.asList(strings));
        return doDeserialize(stringList);
    }

    private TreeNode doDeserialize(List<String> stringList) {
        if (stringList.get(0).equals("None")) {
            stringList.remove(0);
            return null;
        }
        TreeNode node = new TreeNode(Integer.valueOf(stringList.get(0)));
        stringList.remove(0);
        node.left = doDeserialize(stringList);
        node.right = doDeserialize(stringList);
        return node;
    }
}
```















# <a name="Backtracking-Content">Backtracking</a>

## [0077. 组合](https://leetcode-cn.com/problems/combinations/)

直接看代码吧...思路反正就这么个思路，还算清楚，但是限制条件想得有点乱。

```java
//0077. 组合
public class Solution1 {
    private List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        checkCombine(1, n, k, new ArrayList<>());
        return res;
    }
    private void checkCombine(int i, int n, int k, List<Integer> temp) {
    //注意这个限制条件
        if(n-i+temp.size()<k-1){
            return;
        }
        if(k==temp.size()){
            res.add(temp);
            return;
        }else {
            checkCombine(i+1, n, k, new ArrayList<>(temp));
            temp.add(i);
            checkCombine(i+1, n, k, new ArrayList<>(temp));
        }
        return;
    }
}
```

i指的是从1开始递增，n就是题目给定n，k就是题目给定k，同时再传一个List用来记录满足题目要求的集合。

退出条件就是n-i+1+temp.size()<k（这个+1很关键，因为在遍历到i的时候，当前值还没有传入集合，也就是说，此时还是第i-1步），意思是剩余遍历次数加上集合目前大小已经小于规定k时，那么这个集合一定满足不了大小等于k。



## [0216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

```java
class Solution {
	List<List<Integer>> result = new ArrayList<>();
	LinkedList<Integer> path = new LinkedList<>();
	public List<List<Integer>> combinationSum3(int k, int n) {
		backTracking(n, k, 1, 0);
		return result;
	}
	private void backTracking(int targetSum, int k, int startIndex, int sum) {
		// 减枝
		if (sum > targetSum) {
			return;
		}
		if (path.size() == k) {
			if (sum == targetSum) result.add(new ArrayList<>(path));
			return;
		}
		// 减枝 9 - (k - path.size()) + 1
		for (int i = startIndex; i <= 9 - (k - path.size()) + 1; i++) {
			path.add(i);
			sum += i;
			backTracking(targetSum, k, i + 1, sum);
			//回溯
			path.removeLast();
			//回溯
			sum -= i;
		}
	}
}
```



## [0017. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

首先定义一个字符串数组，分别给0-9定义10个字符串，对应题目中的要求。

这题与之前的组合不一样，之前的0077和0216都是选与不选中做选择，而本题是多个选项中选一个，不存在不选的情况。

```java
//0017. 电话号码的字母组合
public class Solution3 {
    List<String> res = new ArrayList<>();
//    StringBuilder sb = new StringBuilder();
    final String[] letterMap = new String[]{"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    public List<String> letterCombinations(String digits) {
        if(digits==null||digits.length()==0){
            return new ArrayList<>();
        }
        checkCombine(digits, 0, new StringBuilder());
        return res;
    }

    private void checkCombine(String digits, int start, StringBuilder sb) {
        if(start>=digits.length()){
            res.add(sb.toString());
            return;
        }
        int index = (int)digits.charAt(start)-48;
        for(int j=0; j<letterMap[index].length(); j++){
            sb.append(letterMap[index].charAt(j));
            checkCombine(digits, start+1, new StringBuilder(sb));
            sb.deleteCharAt(sb.length()-1);
        }
    }
}
```



## [0039. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

第一种方法，效率较低，思路就是有两种走法，要么选取当前元素（因为可以重复选取，所以如果选取当前元素，则依旧把当前index传递下去），要么不选当前元素。但是不需要对candidates进行重新排序，有点暴力的感觉

```java
private void checkCombine(int index, int sum, int[] candidates, int target, List<Integer> list) {
        if(index>=candidates.length||sum>target){
            return;
        }
        if(sum==target){
            res.add(list);
            return;
        }
        checkCombine(index+1, sum, candidates, target, new ArrayList<>(list));
        list.add(candidates[index]);
        checkCombine(index, sum+candidates[index], candidates, target, new ArrayList<>(list));
    }
```

第二种方法，首先要先对数组进行排序，从小到大，因为用for这种方法有个强制终止break，**如果数组不是有序的，那么就有可能会错过下标靠后的较小元素，因为break之后就不会遍历到他们了**，而这个break也是一个非常强大的“剪枝”工具，和自己写的第一种方法相比，可以过滤更多的不必要计算。

```java
public void backtracking(List<Integer> path, int[] candidates, int target, int sum, int index){
        if(sum==target){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=index; i<candidates.length; i++){
            if(sum+candidates[i]>target){
                break;
            }
            path.add(candidates[i]);
            backtracking(path, candidates, target, sum+candidates[i], i);
            path.remove(path.size()-1);
        }
    }
```





## [0040. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

```java
//0040. 组合总和 II
public class Solution5 {
    private List<List<Integer>> res = new ArrayList<>();
    private int temp=-1;
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        backtracking(new ArrayList<>(), candidates, target, 0, 0);
        return res;
    }
    public void backtracking(List<Integer> path, int[] candidates, int target, int sum, int index){
        if(sum==target){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=index; i<candidates.length; i++){
            if(sum+candidates[i]>target){
                break;
            }
            if(candidates[i]==temp){
                continue;
            }
            path.add(candidates[i]);
            backtracking(path, candidates, target, sum+candidates[i], i+1);
            path.remove(path.size()-1);
            temp = candidates[i];
        }
    }
}
```

这题与0039相比，区别在于，0039的数组中元素不重复，但是可以重复选取；0040中数组元素可以重复，但是不能重复选取。

逻辑可以按照0039的来，只要在递推时，把index改为i+1传入即可，因为不能重复选取（0039中传入的就是i本身）。

改完后测试会发现如[1,1,2,3]找target为3，会输出两次[1,2]。原因是递推中传入了第一个1，而代码顺序进行下去后，循环到了第二个1，两条分支都会走到下标为2的val为2的地方，此时都是3，所以输出了两次。

为了避免这种情况，经过思考，在遍历过程中，能够重复选取同一种元素的条件是必须在该元素出现的第一次就选取，不然就不能选取，意思是举个例子[1,1,1,1,4,5]，在这个数组中，如果想选第二个1，或者是第三个1，亦或者是第四个1，你就必须走选取了第一个1的分支，否则就无法选择。

代码实现：

在for循环的末尾位置加一个把candidates[i]赋值给temp的操作，用temp记录下当前candidates值，因为既然能走到这一步，说明并没有走递推，也就是说没有选取当前元素，那么之后遍历下都不能选取该元素。然后for循环的终止条件判断后新加一个判断，判断当前candidates值是否等于temp，等于就continue跳过。



## [0131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

```java
//0131. 分割回文串
public class Solution6 {
    private List<List<String>> res = new ArrayList<>();
    public List<List<String>> partition(String s) {
        backtracking(s, 0, 0, new ArrayList<>());
        return res;
    }
    private void backtracking(String s, int start, int end, List<String> list) {
        if(end>=s.length()){
            return;
        }
        if(isSymmetry(s.substring(start, end+1))){
            list.add(s.substring(start, end+1));
            if(end+1==s.length()){
                res.add(new ArrayList<>(list));
            }
            backtracking(s, end+1, end+1, new ArrayList<>(list));
            list.remove(list.size()-1);
            backtracking(s, start, end+1, new ArrayList<>(list));
        }else {
            backtracking(s, start, end+1, new ArrayList<>(list));
        }
    }
    private Boolean isSymmetry(String s){
        if(s.length()==1){
            return true;
        }
        Queue<Character> queue = new LinkedList<>();
        for(int i=0; i<s.length()/2; i++){
            queue.offer(s.charAt(i));
        }
        for(int i=s.length()-1; i>=s.length()-s.length()/2; i--){
            if(queue.poll()!=s.charAt(i)){
                return false;
            }
        }
        return true;
    }
		//for循环方法
    private void backtracking2(String s, int start, int end, List<String> list){
        if(end>=s.length()){
            return;
        }
        for(int i=end; i<s.length(); i++){
            if(isSymmetry(s.substring(start, end+1))){
                list.add(s.substring(start, end+1));
                backtracking2(s, end+1, end+1, new ArrayList<>(list));
                list.remove(list.size()-1);
            }
        }
        return;
    }
}
```

![131.分割回文串](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f3133312e2545352538382538362545352538392542322545352539422539452545362539362538372545342542382542322e6a7067.png)

跟着这个图的逻辑写得以上第二部分代码



## [0093. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

```java
//0093. 复原 IP 地址
public class Solution7 {
    private List<String> res = new ArrayList<>();
    public List<String> restoreIpAddresses(String s) {
        backtracking(s, 0, new ArrayList());
        return res;
    }
    private void backtracking(String s, int start, List<String> list) {
        if(start>=s.length()||list.size()>4){
            if(list.size()==4){
                res.add(integration(list));
            }
            return;
        }
        StringBuilder sb = new StringBuilder();
        if(s.charAt(start)=='0'){
            sb.append(s.charAt(start));
            list.add(sb.toString());
            backtracking(s, start+1, list);
            list.remove(list.size()-1);
            return;
        }
        for(int i=start; i<s.length(); i++){
            sb.append(s.charAt(i));
            if(Integer.valueOf(sb.toString()).intValue()>255){
                return;
            }
            list.add(sb.toString());
            backtracking(s, i+1, list);
            list.remove(list.size()-1);
        }
    }
    private String integration(List<String> list) {
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<list.size(); i++){
            sb.append(list.get(i));
            if(i<list.size()-1){
                sb.append('.');
            }
        }
        return sb.toString();
    }
}
```

思路：

本题首先有几个限定条件：

1. 这个字符串只能由四个整数构成
2. 每个整数必须位于[0,255]
3. 整数不能有前导0

本题我使用了一个`list<String>`来存放当前已生成的字符串，再遍历完全后，再把list中所有元素拼接到一起，这样可以方便来判断递归生成的结果是否合法。比如对于**<u>条件1</u>**，若递推传入的list大小大于4时，则直接return。对于**<u>条件2</u>**，在吧新的字符串传入递推前，先判断这个字符串中的整数大小是否小于等于255，若大于255则直接return。对于**<u>条件3</u>**，在for循环前加入一个判断，判断是否当前index的字符是'0'，如果是0则直接传入递推。



## [0078. 子集](https://leetcode-cn.com/problems/subsets/)

```java
//0078. 子集
public class Solution8 {
    private List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        backtracking(nums, 0, new ArrayList<>());
        return res;
    }
    private void backtracking(int[] nums, int index, List<Integer> list) {
        if(index>=nums.length){
            
            return;
        }
        list.add(nums[index]);
        backtracking(nums, index+1, list);
        list.remove(list.size()-1);
        backtracking(nums, index+1, list);
    }
}
```

很简单一题，没什么好说的。



## [0090. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

```java
//0090. 子集 II
public class Solution9 {
    private List<List<Integer>> res = new ArrayList<>();
    private int temp = -11;
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        backtracking(nums, 0, new ArrayList<>());
        return res;
    }

    private void backtracking(int[] nums, int start, List<Integer> list) {
        if(start>=nums.length){
            res.add(new ArrayList<>(list));
            return;
        }
        for(int i=start; i<nums.length; i++){
            if(temp==nums[i]){
                continue;
            }
            list.add(nums[i]);
            backtracking(nums, i+1, list);
            list.remove(list.size()-1);
            temp=nums[i];
        }
        res.add(new ArrayList<>(list));
    }
}
```

本题和0040的基础版本，因为都是给定了一个可能存在重复元素的数组，如果把每一种情况都考虑进去，一定会存在重复子集的情况，所以如果是没走递归，走的循环的，走循环即意味着不把当前元素存入list，那么若当前元素和其上一个元素相同，就要跳过循环，否则会和走递归的情况产生重复的结果。

**用此方法，本题一定要先对数组进行排序**



## [0491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/)（需巩固）

![491. 递增子序列1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132343230303232393832342e706e67.png)

在[90.子集II](https://programmercarl.com/0090.子集II.html)中我们是通过排序，再加一个标记数组来达到去重的目的。

而本题求自增子序列，是不能对原数组经行排序的，排完序的数组都是自增子序列了。

**所以不能使用之前的去重逻辑！**

根据上图逻辑，既然不能排序去重，那么就在每次递归中判断是否有选取到过相同元素

```java
//0491. 递增子序列
public class Solution10 {
    private List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> findSubsequences(int[] nums) {
        backtracking(nums, 0, new ArrayList<>(), -101);
        return res;
    }

    private void backtracking(int[] nums, int start, List<Integer> list, int pre) {
        if(start>=nums.length){
            if(list.size()>=2){
                res.add(new ArrayList<>(list));
            }
            return;
        }
        Set<Integer> set = new HashSet<>();
        for(int i=start; i<nums.length; i++){
            if(nums[i]<pre){
                continue;
            }
            if(!set.contains(nums[i])){
                set.add(nums[i]);
                list.add(nums[i]);
                backtracking(nums, i+1, list, nums[i]);
                list.remove(list.size()-1);
            }
        }
        if(list.size()>=2){
            res.add(new ArrayList<>(list));
        }
    }
}
```

此题给出了一个非排序去重的办法，就是在递归中不能选取相同的元素。



## [0046. 全排列](https://leetcode-cn.com/problems/permutations/)

![46.全排列](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f636f64652d7468696e6b696e672d313235333835353039332e66696c652e6d7971636c6f75642e636f6d2f706963732f32303231313032373138313730362e706e67.png)

思路与这张图一样，每次从给定集合中抽选出一个数，然后将剩余元素的集合传入到递归中，但由于题目给的集合是数组，删除元素相对比较困难，所以重新定义一个ArrayList，把数组元素拷贝进去。代码如下：

```java
//0046. 全排列
public class Solution11 {
    private List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        List<Integer> newNums = new ArrayList<>();
        for (int num : nums) {
            newNums.add(num);
        }
        backtracking(newNums, new ArrayList<>());
        return res;
    }

    private void backtracking(List<Integer> list, List<Integer> construct) {
        if(list.size()==0){
            res.add(new ArrayList<>(construct));
            return;
        }
        for(int i=0; i<list.size(); i++){
            construct.add(list.get(i));
            List<Integer> temp = new ArrayList<>(list);
            temp.remove(i);
            backtracking(temp, construct);
            construct.remove(construct.size()-1);
        }
    }
}
```



## [0047. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

![47.全排列II1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132343230313333313232332e706e67.png)

```
//0047. 全排列 II
public class Solution12 {
    private List<List<Integer>> res = new ArrayList<>();
    private int pre = -11;
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<Integer> newNums = new ArrayList<>();
        for (int num : nums) {
            newNums.add(num);
        }
        backtracking(newNums, new ArrayList<>());
        return res;
    }
    private void backtracking(List<Integer> list, List<Integer> construct) {
        if(list.size()==0){
            res.add(new ArrayList<>(construct));
            return;
        }
        for(int i=0; i<list.size(); i++){
            if(list.get(i)==pre){
                continue;
            }
            construct.add(list.get(i));
            List<Integer> temp = new ArrayList<>(list);
            temp.remove(i);
            backtracking(temp, construct);
            construct.remove(construct.size()-1);
            pre = list.get(i);
        }
    }
}
```

因为本题可以排序，所以先对nums排序，利用排序进行去重，也可以用set来去重，但是不知道为啥两者用时差不多。



## [0332. 重新安排行程](https://leetcode-cn.com/problems/reconstruct-itinerary/)

```java
//0332. 重新安排行程
public class Solution13 {
    private List<List<String>> res = new ArrayList<>();
    public List<String> findItinerary(List<List<String>> tickets) {
        backtracking(tickets, tickets.size(), "JFK", 0, new ArrayList<>());
        Collections.sort(res, new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                if(o1.size()==o2.size()){
                    for(int i=0; i<o1.size(); i++){
                        if(o1.get(i).compareTo(o2.get(i))==0){
                            continue;
                        }
                        return o1.get(i).compareTo(o2.get(i));
                    }
                }
                return 0;
            }
        });
        return res.get(0);
    }
    private void backtracking(List<List<String>> tickets, int size, String from, int count, List<String> list) {
        if(count==size){
            //在遍历完全后把终点加进去
            list.add(from);
            res.add(new ArrayList<>(list));
            list.remove(list.size()-1);
            return;
        }
        for(int i=0; i<tickets.size(); i++){
            //用equals去作比较，==比较的是地址
            if(tickets.get(i).get(0).equals(from)){
                List<List<String>> temp = new ArrayList<>(tickets);
                temp.remove(i);
                list.add(tickets.get(i).get(0));
                backtracking(temp, size, tickets.get(i).get(1), count+1, list);
                list.remove(list.size()-1);
            }
        }
    }
}
```

以上代码结果是正确的，但是运行超时，因为没有剪枝，并且排序上也花费了大量时间，思路就是先把所有可能全部罗列出来，然后根据题目要求对列表排序。

```java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> findItinerary(List<List<String>> tickets) {
        Collections.sort(tickets, new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                if(o1.size()==o2.size()){
                    for(int i=0; i<o1.size(); i++){
                        if(o1.get(i).compareTo(o2.get(i))==0){
                            continue;
                        }
                        return o1.get(i).compareTo(o2.get(i));
                    }
                }
                return 0;
            }
        });
        backtracking(tickets, tickets.size(), "JFK", 0, new ArrayList<>());
        return res;
    }
    private void backtracking(List<List<String>> tickets, int size, String from, int count, List<String> list) {
        if(list.size()>size){
            return;
        }
        if(list.size()==size){
            list.add(from);
            res=new ArrayList<>(list);
            return;
        }
        for(int i=0; i<tickets.size(); i++){
            if(tickets.get(i).get(0).equals(from)){
                List<List<String>> temp = new ArrayList<>(tickets);
                temp.remove(i);
                list.add(tickets.get(i).get(0));
                backtracking(temp, size, tickets.get(i).get(1), count+1, list);
                if(list.size()<=size){
                    list.remove(list.size()-1);
                }
            }
        }
    }
}
```

对第一部分代码进行修改，先对tickets列表进行排序，然后只要找到第一个满足题目要求的答案即可，做了少量剪枝，可以通过，但是运行时间排名4.4%，这意味着有大量可以优化的逻辑。

**<u>优化解：</u>**

本题复杂在记录的关系映射，现有两种映射方式`Map<String, Set<String>> hashmap`和`Map<String, Map<String, Integer>> hashmap`两种方式，由于飞机的行程是可以重复的，所以用Set不如用Map直接记录下具体的次数来得方便。所以我们使用`Map<String, Map<String, Integer>> hashmap`更好。

```java
//0332. 重新安排行程（优化写法）
public class Solution14 {
    private Deque<String> res;
    private Map<String, Map<String, Integer>> map;

    private boolean backTracking(int ticketNum){
        if(res.size() == ticketNum + 1){
            return true;
        }
        String last = res.getLast();
        if(map.containsKey(last)){//防止出现null
            for(Map.Entry<String, Integer> target : map.get(last).entrySet()){
                int count = target.getValue();
                if(count > 0){
                    res.add(target.getKey());
                    target.setValue(count - 1);
                    if(backTracking(ticketNum)) return true;
                    res.removeLast();
                    target.setValue(count);
                }
            }
        }
        return false;
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        map = new HashMap<String, Map<String, Integer>>();
        res = new LinkedList<>();
        for(List<String> t : tickets){
            Map<String, Integer> temp;
            if(map.containsKey(t.get(0))){
                temp = map.get(t.get(0));
                temp.put(t.get(1), temp.getOrDefault(t.get(1), 0) + 1);
            }else{
                temp = new TreeMap<>();//升序Map
                temp.put(t.get(1), 1);
            }
            map.put(t.get(0), temp);

        }
        res.add("JFK");
        backTracking(tickets.size());
        return new ArrayList<>(res);
    }
}
```





## [0051. N 皇后](https://leetcode-cn.com/problems/n-queens/)

```java
//0051. N 皇后
public class Solution15 {
    List<List<String>> res = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        Map<Integer, Integer> hashMap = new HashMap<>();
        backtracking(hashMap, 0, n);
        return res;
    }

    private void backtracking(Map<Integer, Integer> hashMap, int rowIndex, int n) {
        if(hashMap.size()==n){
            List<String> resString = createList(hashMap);
            res.add(resString);
            return;
        }
        for(int i=0; i<n; i++){
            if(!hashMap.containsKey(i)){
                Boolean canEat = false;
                Iterator<Map.Entry<Integer, Integer>> iterator = hashMap.entrySet().iterator();
                while (iterator.hasNext()){
                    Map.Entry<Integer, Integer> entry = iterator.next();
                    int column = entry.getKey();
                    int row = entry.getValue();
                    if(rowIndex-row==Math.abs(i-column)){
                        canEat = true;
                        break;
                    }
                }
                if(!canEat){
                    hashMap.put(i, rowIndex);
                    backtracking(hashMap, rowIndex+1, n);
                    hashMap.remove(i);
                }
            }
        }
    }

    private List<String> createList(Map<Integer, Integer> hashMap) {
        List<String> resString = new ArrayList<>();
        List<Map.Entry<Integer, Integer>> mapToList = new ArrayList<>(hashMap.entrySet());
        Collections.sort(mapToList, new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o1.getValue()-o2.getValue();
            }
        });
        Iterator<Map.Entry<Integer, Integer>> iterator = mapToList.iterator();
        while (iterator.hasNext()){
            StringBuilder sb = new StringBuilder();
            Map.Entry<Integer, Integer> entry = iterator.next();
            int column = entry.getKey();
            int row = entry.getValue();
            for(int i=0; i<column; i++){
                sb.append(".");
            }
            sb.append("Q");
            for(int i=column+1; i<hashMap.size(); i++){
                sb.append(".");
            }
            resString.add(sb.toString());
        }
        return resString;
    }
}
```

本人思路是用hashMap来实现，利用Map的Key来记录对应的列，用Map的Value来记录对应的行。因为皇后可以直线走无限格，所以每行每列都至多有一个皇后，而因为在n*n棋盘上有n个皇后，这意味着每行每列都一定有一个皇后。所以用`Map.containsKey()`来判断相同列是否有两个皇后，然后利用递归来保证每行只有一个皇后，即传入递归中的行每次+1。最后因为皇后可以斜45度走无限格，所以用`rowIndex-row==Math.abs(i-column)`来判断斜45度任意角度是否有两个皇后。

与网上思路一样：

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303133303138323533323330332e6a7067.png" alt="51.N皇后" style="zoom:50%;" />

运行速度没那没快的原因，应该出在createList上，我是先构造完map，再把map排序后，再转化成list输出。

```java
class Solution {
    List<List<String>> res = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        char[][] chessboard = new char[n][n];
        for (char[] c : chessboard) {
            Arrays.fill(c, '.');
        }
        backTrack(n, 0, chessboard);
        return res;
    }

    public void backTrack(int n, int row, char[][] chessboard) {
        if (row == n) {
            res.add(Array2List(chessboard));
            return;
        }

        for (int col = 0;col < n; ++col) {
            if (isValid (row, col, n, chessboard)) {
                chessboard[row][col] = 'Q';
                backTrack(n, row+1, chessboard);
                chessboard[row][col] = '.';
            }
        }

    }

    public List Array2List(char[][] chessboard) {
        List<String> list = new ArrayList<>();

        for (char[] c : chessboard) {
            list.add(String.copyValueOf(c));
        }
        return list;
    }


    public boolean isValid(int row, int col, int n, char[][] chessboard) {
        // 检查列
        for (int i=0; i<row; ++i) { // 相当于剪枝
            if (chessboard[i][col] == 'Q') {
                return false;
            }
        }

        // 检查45度对角线
        for (int i=row-1, j=col-1; i>=0 && j>=0; i--, j--) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }

        // 检查135度对角线
        for (int i=row-1, j=col+1; i>=0 && j<=n-1; i--, j++) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
}
```





## [0037. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

```java
public void solveSudoku(char[][] board) {
        solveSudokuHelper(board);
    }

    private boolean solveSudokuHelper(char[][] board){
        //「一个for循环遍历棋盘的行，一个for循环遍历棋盘的列，
        // 一行一列确定下来之后，递归遍历这个位置放9个数字的可能性！」
        for (int i = 0; i < 9; i++){ // 遍历行
            for (int j = 0; j < 9; j++){ // 遍历列
                if (board[i][j] != '.'){ // 跳过原始数字
                    continue;
                }
                for (char k = '1'; k <= '9'; k++){ // (i, j) 这个位置放k是否合适
                    if (isValidSudoku(i, j, k, board)){
                        board[i][j] = k;
                        if (solveSudokuHelper(board)){ // 如果找到合适一组立刻返回
                            return true;
                        }
                        board[i][j] = '.';
                    }
                }
                // 9个数都试完了，都不行，那么就返回false
                return false;
                // 因为如果一行一列确定下来了，这里尝试了9个数都不行，说明这个棋盘找不到解决数独问题的解！
                // 那么会直接返回， 「这也就是为什么没有终止条件也不会永远填不满棋盘而无限递归下去！」
            }
        }
        // 遍历完没有返回false，说明找到了合适棋盘位置了
        return true;
    }

    /**
     * 判断棋盘是否合法有如下三个维度:
     *     同行是否重复
     *     同列是否重复
     *     9宫格里是否重复
     */
    private boolean isValidSudoku(int row, int col, char val, char[][] board){
        // 同行是否重复
        for (int i = 0; i < 9; i++){
            if (board[row][i] == val){
                return false;
            }
        }
        // 同列是否重复
        for (int j = 0; j < 9; j++){
            if (board[j][col] == val){
                return false;
            }
        }
        // 9宫格里是否重复
        int startRow = (row / 3) * 3;
        int startCol = (col / 3) * 3;
        for (int i = startRow; i < startRow + 3; i++){
            for (int j = startCol; j < startCol + 3; j++){
                if (board[i][j] == val){
                    return false;
                }
            }
        }
        return true;
    }
```

逻辑：

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f323032303131313732303435313739302e706e67.png" alt="37.解数独" style="zoom:50%;" />





## [0022. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

```java
	List<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        checkParenthesis(n, new String(), new LinkedList<>());
        return res;
    }

    private void checkParenthesis(int n, String s, Deque<Character> deque) {
        if (deque.size() > n) {
            return;
        }
        if (n == 0) {
            res.add(s);
            return;
        }
        StringBuilder sb = new StringBuilder(s);
        deque.addLast('(');
        checkParenthesis(n, sb.append('(').toString(), new LinkedList<>(deque));
        deque.pollLast();
        if (deque.size() > 0) {
            deque.pollLast();
            sb.deleteCharAt(sb.length() - 1);
            checkParenthesis(n - 1, sb.append(')').toString(), new LinkedList<>(deque));
        }
    }
```

本题最核心的问题在于，如果有用一个队列deque来维护括号生成，那么deque中只能存在'()'左括号，而不能有')'右括号，因为每次有右括号进来，它就需要去抵消掉一个左括号。本代码后续优化其实可以不需要每次都传入一个deque，只需要用一个int整型来记录此时deque中的左括号数量即可。

优化后版本:

```java
	List<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        checkParenthesis(n, 0, new String());
        return res;
    }
    private void checkParenthesis(int n, int size, String s) {
        if (size > n || size < 0) {
            return;
        }
        if (n == 0) {
            res.add(s);
            return;
        }
        StringBuilder sb = new StringBuilder(s);
        checkParenthesis(n, size + 1, sb.append('(').toString());
        sb.deleteCharAt(sb.length() - 1);
        checkParenthesis(n - 1, size - 1, sb.append(')').toString());
    }
```

1. 当deque的大小大于n时，说明左括号数量超过了字符串的一半，不满足题目要求，直接return

2. 当deque的大小小于0时，说明多了一个右括号，不满足题目要求，直接return

3. 两个分支：

   1. 字符串中插入左括号

      `checkParenthesis(n, size, sb.append('(').toString());`

   2. 字符串中插入右括号

      `checkParenthesis(n - 1, size - 1, sb.append(')').toString());`









## [0869. 重新排序得到 2 的幂](https://leetcode.cn/problems/reordered-power-of-2/)

```java
static Set<Integer> set = new HashSet<>();
static {
    for (int i = 1; i < (int) 1e9; i *= 2) set.add(i);
}
int[] cnts = new int[10];//用于存储0-9出现的个数
public boolean reorderedPowerOf2(int n) {
    int digit = 0;
    while (n != 0) {
        cnts[n % 10]++;
        n /= 10;
        digit++;
    }
    return findPowerOf2(digit, 0, 0);
}

private Boolean findPowerOf2(int digit, int index, int sum) {
    if (index == digit) {
        return set.contains(sum);
    }
    for (int i = 0; i < 10; i++) {
        if (cnts[i] != 0) {
            cnts[i]--;
            if ((i != 0 || sum != 0) && findPowerOf2(digit, index + 1, sum * 10 + i)) return true;
            cnts[i]++;
        }
    }
    return false;
}
```

1. 先提前将[1,1e9]内的所有2的幂给预处理放入set中
2. 利用DFS对n进行重排
3. 每次重排完成，判断该值是否出现在set中，若出现则返回true，反之亦然



方法二：

词频统计，因为预处理[1,1e9]内的所有2的幂后发现，一共也就只有30个2的幂，因此我们可以提前将这构成30个数的词频组合统计出来，并用哈希表记录，利用一个长度为10的数组来统计每一个组合0-9所使用的次数。

```java
class Solution {
    Set<String> powerOf2Digits = new HashSet<String>();

    public boolean reorderedPowerOf2(int n) {
        init();
        return powerOf2Digits.contains(countDigits(n));
    }

    public void init() {
        for (int n = 1; n <= 1e9; n <<= 1) {
            powerOf2Digits.add(countDigits(n));
        }
    }

    public String countDigits(int n) {
        char[] cnt = new char[10];
        while (n > 0) {
            ++cnt[n % 10];
            n /= 10;
        }
        return new String(cnt);
    }
}
```







# Greedy Algorithm

## [0455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

```java
//0455. 分发饼干
public class Solution1 {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int count = 0;
        int indexS = 0;
        for (int i = 0; i < g.length;) {
            if(indexS >= s.length){
                break;
            }
            if(s[indexS] >= g[i]){
                count++;
                i++;
            }
            indexS++;
        }
        return count;
    }
}
```

思路：

先把两个数组排序，以从小到大顺序排完，此时只要对两个数组进行遍历

- 如果s[j]>=g[i]说明此时找到了最贪心的分配方法，即当前饼干可以分配给某人损失最小。此时indexS++，i++。
- 如果s[j]<g[i]那说明已经这块饼干满足不了任意一个人，所以直接过滤掉，让indexS++，并且i不自增。



## [0376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

![376.摆动序列](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132343137343332373539372e706e67.png)

**局部最优：删除单调坡度上的节点（不包括单调坡度两端的节点），那么这个坡度就可以有两个局部峰值**。

**整体最优：整个序列有最多的局部峰值，从而达到最长摆动序列**。

局部最优推出全局最优，并举不出反例，那么试试贪心！

（为方便表述，以下说的峰值都是指局部峰值）

**实际操作上，其实连删除的操作都不用做，因为题目要求的是最长摆动子序列的长度，所以只需要统计数组的峰值数量就可以了（相当于是删除单一坡度上的节点，然后统计长度）**

**这就是贪心所贪的地方，让峰值尽可能的保持峰值，然后删除单一坡度上的节点**。

本题代码实现中，还有一些技巧，例如统计峰值的时候，数组最左面和最右面是最不好统计的。

例如序列[2,5]，它的峰值数量是2，如果靠统计差值来计算峰值个数就需要考虑数组最左面和最右面的特殊情况。

所以可以针对序列[2,5]，可以假设为[2,2,5]，这样它就有坡度了即preDiff = 0，如图：

![376.摆动序列1](/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132343137343335373631322e706e67.png)

```java
//0376. 摆动序列
public class Solution2 {
    public int wiggleMaxLength(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return nums.length;
        }
        //当前差值
        int curDiff = 0;
        //上一个差值
        int preDiff = 0;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            //得到当前差值
            curDiff = nums[i] - nums[i - 1];
            //如果当前差值和上一个差值为一正一负
            //等于0的情况表示初始时的preDiff
            if ((curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0)) {
                count++;
                preDiff = curDiff;
            }
        }
        return count;
    }
}
```



## [0053. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

```java
//0053. 最大子数组和
public class Solution3 {
    public int maxSubArray(int[] nums) {
        int sum = 0;
        int index = 0;
        int max = -10001;
        for(int i=0; i<nums.length; i++){
            sum+=nums[i];
            max = Math.max(max, sum);
          //sum<0意味着重新开始
            if(sum<0){
                sum=0;
            }
        }
        return max;
    }
}
```

题目给定-10^4<=nums[i]<=10^4，所以先设定max值为-10001，防止题目给一个类似这样的输入条件[-10000]出错。

本题是要找连续的子数组让总和最大，可以用一个变量来实时记录从某一位置到某一位置的总和，如果此变量值变为负数，就重新开始记录新的子数组，并记录之前子数组中的最大和。

**局部最优**：当前“连续和”为负数的时候立刻放弃，从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会越来越小。

**全局最优**：选取最大“连续和”

![53.最大子序和](/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f676966732f35332e2545362539432538302545352541342541372545352541442539302545352542412538462545352539322538432e676966.png)





## [0122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```java
//0122. 买卖股票的最佳时机 II
public class Solution4 {
    public int maxProfit(int[] prices) {
        if(prices.length<2){
            return 0;
        }
        Boolean isBuy = false;
        int sum=0;
        for(int i=1; i<prices.length; i++){
            if(!isBuy){
                if(prices[i]-prices[i-1]>0){
                    sum-=prices[i-1];
                    isBuy=true;
                }
            }else {
                if(prices[i]-prices[i-1]<=0){
                    sum+=prices[i-1];
                    isBuy=false;
                }
            }
        }
        if(isBuy){
            sum+=prices[prices.length-1];
        }
        return sum;
    }
}
```

局部最佳：

- 如果还没有买股票，则prices[i]<prices[i-1]意味着直接跳过i-1，不买i-1这个股票
- 如果此时买了股票，则prices[i]<=prices[i-1]就直接抛出当前彩票，在等于的情况也要抛出，因为这样可以让利益最大化，又多了一次可以买入的机会。否则就不卖出股票继续遍历。

全局最佳：

每次买进卖出彩票都是赚的最多的。



## [0055. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

```java
//0055. 跳跃游戏
public class Solution5 {
    public boolean canJump(int[] nums) {
        if(nums.length==1){
            return true;
        }
      //下面循环i是从1开始的，所以要对下标为0元素单独做一个判断
        if(nums[0]==0){
            return false;
        }
        int temp = 0;
        for(int i=1; i<nums.length; i++){
            if(nums[temp]+temp>=nums.length-1){
                return true;
            }
            if(nums[i]==0){
                if(temp+nums[temp]>i){
                    continue;
                }else {
                    return false;
                }
            }else {
                if(nums[i]+i-temp>nums[temp]){
                    temp=i;
                }
            }
        }
        return true;
    }
}
```

**贪心算法局部最优解：每次取最大跳跃步数（取最大覆盖范围)**

**整体最优解：最后得到整体最大覆盖范围，看是否能到终点**。

<img src="/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132343135343735383232392e706e67.png" alt="55.跳跃游戏" style="zoom: 67%;" />





## [0045. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

```java
//0045. 跳跃游戏 II
public class Solution6 {
    public int jump(int[] nums) {
        if(nums.length==1){
            return 0;
        }
        int count = 0;
        int temp = 0;
        for(int i=0; i<nums.length;){
            int max = -1;
            if(i+nums[i]>=nums.length-1){
                count++;
                break;
            }
            for(int j=1; j<=nums[i]; j++){
                if(nums[i+j]+j>=max){
                    max=nums[i+j]+j;
                    temp=i+j;
                }
            }
            count++;
            i=temp;
        }
        return count;
    }
}
```

本题和0055跳跃游戏不一样的地方在于，本题首先保证了可以到达数组的最后一个位置的条件，然后要求跳跃最小次数。

局部最优：选取在能跳到的范围内所有位置中，下一步可以跳到最远位置的那个位置。

<img src="/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313230313233323330393130332e706e67.png" alt="45.跳跃游戏II" style="zoom:67%;" />





## [1005. K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

```java
//1005. K 次取反后最大化的数组和
public class Solution7 {
    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        for(int i=0; i<nums.length;){
            if(k<1){
                break;
            }
          //如果元素值小于0，则继续遍历，把尽可能多的负数变为正数
            if(nums[i]<0){
                k--;
                nums[i]=-nums[i];
                i++;
            }else {
              //如果k为偶数，则不用再做变动了，一负一正还是正
                if(k%2==0){
                    k=0;
                    break;
                  //反之，把最小的那个数变为负数，此时要判断i是不是0，是0则不需要和前一个数做比较
                }else {
                    if(i==0){
                        nums[i]=-nums[i];
                    }else {
                        if(nums[i-1]<nums[i]){
                            nums[i-1]=-nums[i-1];
                        }else {
                            nums[i]=-nums[i];
                        }
                    }
                    k=0;
                    break;
                }
            }
        }
      //这是为了防止数组全是负数并且k大于数组个数的情况
        if(k>0){
            if(k%2==1){
                nums[nums.length-1]=-nums[nums.length-1];
            }
        }
        int sum = 0;
        for (int num : nums) {
            sum+=num;
        }
        return sum;
    }
}
```

- 第一步：将数组按照绝对值大小从大到小排序，**注意要按照绝对值的大小**
- 第二步：从前向后遍历，遇到负数将其变为正数，同时K--
- 第三步：如果K还大于0，那么反复转变数值最小的元素，将K用完
- 第四步：求和

本人的思路非常混乱，建议参照下面的：

```java
class Solution {
    public int largestSumAfterKNegations(int[] nums, int K) {
    	// 将数组按照绝对值大小从大到小排序，注意要按照绝对值的大小
	nums = IntStream.of(nums)
		     .boxed()
		     .sorted((o1, o2) -> Math.abs(o2) - Math.abs(o1))
		     .mapToInt(Integer::intValue).toArray();
	int len = nums.length;	    
	for (int i = 0; i < len; i++) {
	    //从前向后遍历，遇到负数将其变为正数，同时K--
	    if (nums[i] < 0 && K > 0) {
	    	nums[i] = -nums[i];
	    	K--;
	    }
	}
	// 如果K还大于0，那么反复转变数值最小的元素，将K用完

	if (K % 2 == 1) nums[len - 1] = -nums[len - 1];
	return Arrays.stream(nums).sum();

    }
}
```





## [0134. 加油站](https://leetcode-cn.com/problems/gas-station/)

```java
//0134. 加油站
public class Solution8 {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum=0;
        int start=0;
        int end=0;
        while (end<gas.length){
            //从第一位开始，如果目前总和大于0则可以继续往右走
            if(sum+gas[end]-cost[end]>=0){
                sum+=gas[end]-cost[end];
                end++;
                if(end==start){
                    break;
                }
                //如果总和小于0，则往左走去找能让到达end时总和大于0的情况
            }else {
                if(start==0){
                    start=gas.length-1;
                }else {
                    start--;
                }
                sum+=gas[start]-cost[start];
                if(start==end){
                    break;
                }
            }
        }
        if(sum>=0){
            return start;
        }else {
            return -1;
        }
    }
}
```

全局贪心：

- 情况一：如果gas的总和小于cost总和，那么无论从哪里出发，一定是跑不了一圈的
- 情况二：rest[i] = gas[i]-cost[i]为一天剩下的油，i从0开始计算累加到最后一站，如果累加没有出现负数，说明从0出发，油就没有断过，那么0就是起点。
- 情况三：如果累加的最小值是负数，汽车就要从非0节点出发，从后向前，看哪个节点能这个负数填平，能把这个负数填平的节点就是出发节点。

以上代码是没有进行1和2剪枝的，直接走的情况3，所以运行效率稍微低了一丢丢。







## [0135. 分发糖果](https://leetcode-cn.com/problems/candy/)

```java
//0135. 分发糖果
public class Solution9 {
    public int candy(int[] ratings) {
        int[] candyVec = new int[ratings.length];
        candyVec[0] = 1;
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candyVec[i] = candyVec[i - 1] + 1;
            } else {
                candyVec[i] = 1;
            }
        }

        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candyVec[i] = Math.max(candyVec[i], candyVec[i + 1] + 1);
            }
        }

        int ans = 0;
        for (int s : candyVec) {
            ans += s;
        }
        return ans;
    }
}
```

- 先确定右边评分大于左边的情况（也就是从前向后遍历）

  此时局部最优：只要右边评分比左边大，右边的孩子就多一个糖果，全局最优：相邻的孩子中，评分高的右孩子获得比左边孩子更多的糖果

  局部最优可以推出全局最优。

  如果ratings[i] > ratings[i - 1] 那么[i]的糖 一定要比[i - 1]的糖多一个，所以贪心：candyVec[i] = candyVec[i - 1] + 1

- 再确定左孩子大于右孩子的情况（从后向前遍历）

  如果 ratings[i] > ratings[i + 1]，此时candyVec[i]（第i个小孩的糖果数量）就有两个选择了，一个是candyVec[i + 1] + 1（从右边这个加1得到的糖果数量），一个是candyVec[i]（之前比较右孩子大于左孩子得到的糖果数量）。

  那么又要贪心了，局部最优：取candyVec[i + 1] + 1 和 candyVec[i] 最大的糖果数量，保证第i个小孩的糖果数量即大于左边的也大于右边的。全局最优：相邻的孩子中，评分高的孩子获得更多的糖果。

  局部最优可以推出全局最优。

  所以就取candyVec[i + 1] + 1 和 candyVec[i] 最大的糖果数量，**candyVec[i]只有取最大的才能既保持对左边candyVec[i - 1]的糖果多，也比右边candyVec[i + 1]的糖果多**。





## [0860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

```java
public boolean lemonadeChange(int[] bills) {
        int cash_5 = 0;
        int cash_10 = 0;

        for (int i = 0; i < bills.length; i++) {
            if (bills[i] == 5) {
                cash_5++;
            } else if (bills[i] == 10) {
                cash_5--;
                cash_10++;
            } else if (bills[i] == 20) {
                if (cash_10 > 0) {
                    cash_10--;
                    cash_5--;
                } else {
                    cash_5 -= 3;
                }
            }
            if (cash_5 < 0 || cash_10 < 0) return false;
        }
        
        return true;
    }
```

- 情况一：账单是5，直接收下。
- 情况二：账单是10，消耗一个5，增加一个10
- 情况三：账单是20，优先消耗一个10和一个5，如果不够，再消耗三个5





## [0406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

```java
//0406. 根据身高重建队列
public class Solution11 {
    public int[][] reconstructQueue(int[][] people) {
        int[][] res = new int[people.length][people[0].length];
        for(int i=0; i<res.length; i++){
            res[i][0]=-1;
        }
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[0]==o2[0]){
                    return o1[1]-o2[1];
                }
                return o1[0]-o2[0];
            }
        });
        for(int i=0; i<people.length; i++){
            int pos=people[i][1];
            int count = 0;
            for(int j=0; j<people.length; j++){
                if(res[j][0]!=-1){
                    if(res[j][0]==people[i][0]){
                        count++;
                    }
                    continue;
                }else {
                    if(count==pos){
                        res[j]=people[i];
                        break;
                    }
                    count++;
                }
            }
        }
        return res;
    }
}
```

**本人思路：**

先对people数组进行排序，按照身高优先排序，身高相同再按照前边有多少人的数量来排序。

定义res数组，把res所有元素的第一个元素设置为-1，方便判断，如果遍历到任意子数组的首元素为-1就意味着当前位置还没有人占位。

根据`people[i][1]`的值来遍历，如果已经有元素占位了，并且之前没有和自己相同大小的元素，则count不自增，否则count++（这里的逻辑就显得有些乱，之后看好的解法）。



**优质思路：**

按照身高排序之后，优先按身高高的people的k来插入，后序插入节点也不会影响前面已经插入的节点，最终按照k的规则完成了队列。

所以在按照身高从大到小排序后：

**局部最优：优先按身高高的people的k来插入。插入操作过后的people满足队列属性**

**全局最优：最后都做完插入操作，整个队列满足题目队列属性**

排序完的people： [[7,0], [7,1], [6,1], [5,0], [5,2]，[4,4]]

插入的过程：

- 插入[7,0]：[[7,0]]
- 插入[7,1]：[[7,0],[7,1]]
- 插入[6,1]：[[7,0],[6,1],[7,1]]
- 插入[5,0]：[[5,0],[7,0],[6,1],[7,1]]
- 插入[5,2]：[[5,0],[7,0],[5,2],[6,1],[7,1]]
- 插入[4,4]：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]

相当于从res数组头部开始往后遍历到`index=people[i][1]`时插入即可

```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        // 身高从大到小排（身高相同k小的站前面）
        Arrays.sort(people, (a, b) -> {
            if (a[0] == b[0]) return a[1] - b[1];
            return b[0] - a[0];
        });

        LinkedList<int[]> que = new LinkedList<>();

        for (int[] p : people) {
            que.add(p[1],p);
        }

        return que.toArray(new int[people.length][]);
    }
}
```





## [0452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

```java
//0452. 用最少数量的箭引爆气球
public class Solution12 {
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0]-o2[0];
            }
        });
        int count = 0;
        for(int i=0; i<points.length; i++){
            if(points[i][0]==points[i][1]){
                continue;
            }
            int left = points[i][0];
            int right = points[i][1];
            for(int j=i+1; j<points.length; j++){
                if(points[j][0]==points[j][1]){
                    continue;
                }
                if(points[j][0]>right||points[j][1]<left){
                    continue;
                }else {
                    left = Math.max(points[j][0], left);
                    right = Math.min(points[j][1], right);
                    points[j][0]=points[j][1];
                }
            }
            points[i][0]=points[i][1];
            count++;
        }
        return count;
    }
}
```

将所有已经射爆了的气球的`points[i][0]==points[i][1]`，这样在循环中加入判断条件，如果相等就意味着已经被射爆了，直接跳过。

不知道为什么出问题了出错（**<u>没有排序，为什么要排序？</u>**）。





如何使用最少的弓箭呢？

直觉上来看，貌似只射重叠最多的气球，用的弓箭一定最少，那么有没有当前重叠了三个气球，我射两个，留下一个和后面的一起射这样弓箭用的更少的情况呢？

尝试一下举反例，发现没有这种情况。

那么就试一试贪心吧！局部最优：当气球出现重叠，一起射，所用弓箭最少。全局最优：把所有气球射爆所用弓箭最少。

**算法确定下来了，那么如何模拟气球射爆的过程呢？是在数组中移除元素还是做标记呢？**

如果真实的模拟射气球的过程，应该射一个，气球数组就remove一个元素，这样最直观，毕竟气球被射了。

但仔细思考一下就发现：如果把气球排序之后，从前到后遍历气球，被射过的气球仅仅跳过就行了，没有必要让气球数组remote气球，只要记录一下箭的数量就可以了。

以上为思考过程，已经确定下来使用贪心了，那么开始解题。

**为了让气球尽可能的重叠，需要对数组进行排序**。

那么按照气球起始位置排序，还是按照气球终止位置排序呢？

其实都可以！只不过对应的遍历顺序不同，我就按照气球的起始位置排序了。

既然按照起始位置排序，那么就从前向后遍历气球数组，靠左尽可能让气球重复。

从前向后遍历遇到重叠的气球了怎么办？

**如果气球重叠了，重叠气球中右边边界的最小值 之前的区间一定需要一个弓箭**。

以题目示例： [[10,16],[2,8],[1,6],[7,12]]为例，如图：（方便起见，已经排序）

![452.用最少数量的箭引爆气球](/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313132333130313932393739312e706e67.png)

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        if (points.length == 0) return 0;
        Arrays.sort(points, (o1, o2) -> Integer.compare(o1[0], o2[0]));

        int count = 1;
        for (int i = 1; i < points.length; i++) {
            if (points[i][0] > points[i - 1][1]) {
                count++;
            } else {
                points[i][1] = Math.min(points[i][1],points[i - 1][1]);
            }
        }
        return count;
    }
}
```





## [0435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

```java
//0435. 无重叠区间
public class Solution13 {
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[0]==o2[0]){
                    return o1[1]-o2[1];
                }
                return o1[0]-o2[0];
            }
        });
        int count = 0;
        for(int i=1; i<intervals.length; i++){
            if(intervals[i][0]>=intervals[i-1][1]){
                continue;
            }
            if(intervals[i][1]>intervals[i-1][1]){
                intervals[i]=intervals[i-1];
                count++;
                continue;
            }else {
                count++;
                continue;
            }
        }
        return count;
    }
  	//更好的写法
  	public int eraseOverlapIntervals2(int[][] intervals) {

        Arrays.sort(intervals,(a,b)->{
            return Integer.compare(a[0],b[0]);
        });
        int remove = 0;
        int pre = intervals[0][1];
        for(int i=1;i<intervals.length;i++){
            if(pre>intervals[i][0]) {
                remove++;
                pre = Math.min(pre,intervals[i][1]);
            }
            else pre = intervals[i][1];
        }
        return remove;
    }
}
```

和0452.用最少数量的箭引爆气球一样，这题先按照左边界排序，再按照右边界排序

- 如果`intervals[i][0]>=intervals[i-1][1]`即i的左边界已经大于等于i-1的右边界了，此时满足题目要求，直接continue即可。
- 如果`intervals[i][1]>intervals[i-1][1]`即i的右边界大于i-1，那么果断把右边界大的i移除，移除方式我采用的就是把`intervals[i]=intervals[i-1]`赋值，这样也保证了接下来循环的有效性。
- 如果`intervals[i][1]<=intervals[i-1][1]`即i-1的右边界大于等于i，那么就把右边界大的i-1移除。



**方法二**

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length < 2) return 0;

        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] != o2[1]) {
                    return Integer.compare(o1[1],o2[1]);
                } else {
                    return Integer.compare(o1[0],o2[0]);
                }
            }
        });

        int count = 1;
        int edge = intervals[0][1];
        for (int i = 1; i < intervals.length; i++) {
            if (edge <= intervals[i][0]){
                count ++; //non overlap + 1
                edge = intervals[i][1];
            }
        }
        return intervals.length - count;
    }
}
```

**按照右边界排序，从左向右记录非交叉区间的个数。最后用区间总数减去非交叉区间的个数就是需要移除的区间个数了**。

此时问题就是要求非交叉区间的最大个数。

右边界排序之后，局部最优：优先选右边界小的区间，所以从左向右遍历，留给下一个区间的空间大一些，从而尽量避免交叉。全局最优：选取最多的非交叉区间。

局部最优推出全局最优，试试贪心！

这里记录非交叉区间的个数还是有技巧的，如图：

![435.无重叠区间](/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313232313230313535333631382e706e67.png)

区间，1，2，3，4，5，6都按照右边界排好序。

每次取非交叉区间的时候，都是可右边界最小的来做分割点（这样留给下一个区间的空间就越大），所以第一条分割线就是区间1结束的位置。

接下来就是找大于区间1结束位置的区间，是从区间4开始。**那有同学问了为什么不从区间5开始？别忘已经是按照右边界排序的了**。

区间4结束之后，在找到区间6，所以一共记录非交叉区间的个数是三个。

总共区间个数为6，减去非交叉区间的个数3。移除区间的最小数量就是3。



## [0763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)

```java
//0763. 划分字母区间
public class Solution14 {
    public List<Integer> partitionLabels(String s) {
        List<Integer> res = new ArrayList<>();
        int[] c = new int[26];
        for(int i=0; i<s.length(); i++){
            int index = s.charAt(i) - 97;
            c[index]++;
        }
        Set<Character> set = new HashSet<>();
        int count = 0;
        for(int i=0; i<s.length(); i++){
            c[s.charAt(i)-97]--;
            count++;
            if(!set.contains(s.charAt(i))){
                if(c[s.charAt(i)-97]!=0){
                    set.add(s.charAt(i));
                }
            }else {
                if(c[s.charAt(i)-97]==0){
                    set.remove(s.charAt(i));
                }
            }
            if(set.isEmpty()){
                res.add(count);
                count = 0;
            }
        }
        return res;
    }
}
```

思路：

先把字符串每个字符遍历，提前创建一个26个单位的int数组，用来记录从`a`到`z`这26个字母出现的次数。

创建一个Set集合用来记录从**记录位置开始到遍历到位置中间所探测到的字符集合**，每次循环判断set是否为空，为空说明应该分隔字符串。



**解法二：**

在遍历的过程中相当于是要找每一个字母的边界，**如果找到之前遍历过的所有字母的最远边界，说明这个边界就是分割点了**。此时前面出现过所有字母，最远也就到这个边界了。

可以分为如下两步：

- 统计每一个字符最后出现的位置
- 从头遍历字符，并更新字符的最远出现下标，如果找到字符最远出现位置下标和当前下标相等了，则找到了分割点

如图：

![763.划分字母区间](/Users/qinyang/markdown/LeetCode/图片/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313232323139313932343431372e706e67.png)

```java
class Solution {
    public List<Integer> partitionLabels(String S) {
        List<Integer> list = new LinkedList<>();
        int[] edge = new int[26];
        char[] chars = S.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            edge[chars[i] - 'a'] = i;
        }
        int idx = 0;
        int last = -1;
        for (int i = 0; i < chars.length; i++) {
            idx = Math.max(idx,edge[chars[i] - 'a']);
            if (i == idx) {
                list.add(i - last);
                last = i;
            }
        }
        return list;
    }
}
```







## [0056. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```java
//0056. 合并区间
public class Solution15 {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0]-o2[0];
            }
        });
        List<List<Integer>> res = new ArrayList<>();
        int start = intervals[0][0];
        int end = intervals[0][1];
        for(int i=1; i<intervals.length; i++){
            if(intervals[i][0]>end){
                List<Integer> array = new ArrayList<>();
                array.add(start);
                array.add(end);
                res.add(array);
                start = intervals[i][0];
                end = intervals[i][1];
            }else {
                end = Math.max(end, intervals[i][1]);
            }
        }
        List<Integer> array = new ArrayList<>();
        array.add(start);
        array.add(end);
        res.add(array);
        int[][] res2 = new int[res.size()][2];
        for(int i=0; i<res.size(); i++){
            for(int j=0; j<2; j++){
                res2[i][j] = res.get(i).get(j);
            }
        }
        return res2;
    }
}
```

思路：

1. 先按左边界大小来排序（从小到大）
2. 先记录数组下标为0元素的左边界和右边界
3. 从i=1开始循环判断`if(intervals[i][0]>end)`，这个条件意思就是判断新的区间是否和当前区间有交集：
   - 有交集，把大的右边界赋值给end
   - 无交集，把新区间左边界和右边界分别赋值给start和end，并把当前区间左边界和右边界塞入list
4. 循环结束后要把当前区间左边界和右边界赛入list
5. 重新构造二维数组





## [0738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)

```java
//0738. 单调递增的数字
public class Solution16 {
    public int monotoneIncreasingDigits(int n) {
        if(n<10){
            return n;
        }
        StringBuilder sb = new StringBuilder();
        String newN = String.valueOf(n);
        Boolean isMonotone = true;
        int temp = 0;
        for(int i=1; i<newN.length(); i++){
            String s1 = String.valueOf(newN.charAt(i-1));
            String s2 = String.valueOf(newN.charAt(i));
            if(Integer.parseInt(s2)>Integer.parseInt(s1)){
                temp = i;
                sb.append(newN.charAt(i-1));
                continue;
            }else if(Integer.parseInt(s2)==Integer.parseInt(s1)){
                sb.append(newN.charAt(i-1));
                continue;
            }else {
                sb.delete(temp,sb.length());
                String s3 = String.valueOf(newN.charAt(temp));
                sb.append(Integer.parseInt(s3)-1);
                isMonotone = false;
                break;
            }
        }
        if(isMonotone){
            return n;
        }else {
            for(int i=temp+1; i<newN.length(); i++){
                sb.append("9");
            }
        }
        if(String.valueOf(sb.charAt(0)).equals("0")){
            return Integer.parseInt(sb.substring(1));
        }
        return n;
    }
}
```

思路：

1. 本题首先我是选择了使用把整型转换成字符串来处理
2. temp的作用是，如果出现连续位数相同的情况，记录第一位的下标，如332，最后要变成299，是改变下标为0的3
3. isMonotone的作用是判断n是否为单调递增
4. 对字符串遍历，即对n从高位开始往低位遍历，此时有三种情况：
   - `Integer.parseInt(s2)>Integer.parseInt(s1)`，即高位小于低位，此时正常把高位放入StringBuilder中，并更新temp。
   - `Integer.parseInt(s2)==Integer.parseInt(s1)`，即高位等于低位，此时把高位放入StringBuilder，但不更新temp，让temp保持第一位的下标。
   - `Integer.parseInt(s2)<Integer.parseInt(s1)`，即高位小于低位，此时把高位-1后放入StringBuilder中，并把isMonotone赋值为false。
5. 退出循环后，判断isMonotone是否为真：
   - 如果是真说明n本身就是单调递增的，直接返回n即可。
   - 如果非真，往StringBuilder后继续添加`newN.length()-temp-2`个"9"即可。
6. 判断第一位数是否为0，为0就重新截取字符串。



**解法二：**

**局部最优：遇到strNum[i - 1] > strNum[i]的情况，让strNum[i - 1]--，然后strNum[i]给为9，可以保证这两位变成最大单调递增整数**。

**全局最优：得到小于等于N的最大单调递增的整数**。

**但这里局部最优推出全局最优，还需要其他条件，即遍历顺序，和标记从哪一位开始统一改成9**。

此时是从前向后遍历还是从后向前遍历呢？

从前向后遍历的话，遇到strNum[i - 1] > strNum[i]的情况，让strNum[i - 1]减一，但此时如果strNum[i - 1]减一了，可能又小于strNum[i - 2]。

这么说有点抽象，举个例子，数字：332，从前向后遍历的话，那么就把变成了329，此时2又小于了第一位的3了，真正的结果应该是299。

**==所以从前后向遍历会改变已经遍历过的结果！==**

那么从后向前遍历，就可以重复利用上次比较得出的结果了，从后向前遍历332的数值变化为：332 -> 329 -> 299

```java
class Solution {
    public int monotoneIncreasingDigits(int N) {
        String[] strings = (N + "").split("");
        int start = strings.length;
        for (int i = strings.length - 1; i > 0; i--) {
            if (Integer.parseInt(strings[i]) < Integer.parseInt(strings[i - 1])) {
                strings[i - 1] = (Integer.parseInt(strings[i - 1]) - 1) + "";
                start = i;
            }
        }
        for (int i = start; i < strings.length; i++) {
            strings[i] = "9";
        }
        return Integer.parseInt(String.join("",strings));
    }
}
```

关于`String[] strings = (N + "").split("")`

```java
int n = 332;
        String[] strings = (n+"").split("");
        System.out.println(n+"");
        for (String string : strings) {
            System.out.println(string);
        }
```

输出结果为：

```java
332
3
3
2
```

==`split("")`的功能相当于把字符串每个字符隔开来==







## [0714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)（未能AC）

如果使用贪心策略，就是最低值买，最高值（如果算上手续费还盈利）就卖。

此时无非就是要找到两个点，买入日期，和卖出日期。

- 买入日期：其实很好想，遇到更低点就记录一下。
- 卖出日期：这个就不好算了，但也没有必要算出准确的卖出日期，只要当前价格大于（最低价格+手续费），就可以收获利润，至于准确的卖出日期，就是连续收获利润区间里的最后一天（并不需要计算是具体哪一天）。

所以我们在做收获利润操作的时候其实有三种情况：

- 情况一：收获利润的这一天并不是收获利润区间里的最后一天（不是真正的卖出，相当于持有股票），所以后面要继续收获利润。
- 情况二：前一天是收获利润区间里的最后一天（相当于真正的卖出了），今天要重新记录最小价格了。
- 情况三：不作操作，保持原有状态（买入，卖出，不买不卖）

```java
// 贪心思路
	public int maxProfit(int[] prices, int fee) {
        int result = 0;
        int minPrice = prices[0]; // 记录最低价格
        for (int i = 1; i < prices.length; i++) {
            // 情况二：相当于买入
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            }

            // 情况三：保持原有状态（因为此时买则不便宜，卖则亏本）
            if (prices[i] >= minPrice && prices[i] <= minPrice + fee) {
                continue;
            }

            // 计算利润，可能有多次计算利润，最后一次计算利润才是真正意义的卖出
            if (prices[i] > minPrice + fee) {
                result += prices[i] - minPrice - fee;
                minPrice = prices[i] - fee; // 情况一，这一步很关键，只有当差值大于fee时，才有可能让这次卖出一定是赚的
            }
        }
        return result;
    }
```

实际上在初始化时`int minPrice = prices[0]`就默认开始即买入股票，然后开始逻辑判断：

- 手上无论何时都有一支股票，如果没有卖出，那么只要下一个prices元素（第二天的股票）比当前手上股票价格要低，那么果断换成不买当前股票而转买第二天的股票。

  ```java
  			if (prices[i] < minPrice) {
                  minPrice = prices[i];
              }
  ```

- 当第二天股票价格比今天手持股票高，那么首先不会考虑替换股票了，再者就是判断第二天如果是打算卖出的话（因为已经存在利润了，可是因为有手续费的存在），所以要判断第二天股票价格与今天股票价格的差值是否大于手续费，不然还是亏的。这两个条件是一个“和”的判断，如果满足则保持原有状态。

- 如果第二天股票价格与今天股票的价格差值大于手续费，那么就可以卖出股票，并且要更新最小价格，因为卖出的时候就要考虑下一笔交易手续费的问题，==所以`minPrice = prices[i] - fee`==（第一次交易不管怎么样都要交手续费，不存在和之前一次交易做利润比较，所以初始化时minPrice就是prices[i]）。

不用数学公式推，感觉非常难直接想到这个简化`minPrice = prices[i] - fee`，这个代码巧妙在它同时适应了两种情况，如果不需要分段，那么在减去fee的费用实际上在之后的公式里，`prices[i]-minPrice=prices[i]-(prices[pre]-fee)=prices[i]-prices[pre]+fee`。







## [0968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/)

```java
//0968. 监控二叉树
public class Solution18 {
    private int count = 0;
    public int minCameraCover(TreeNode root) {
        checkMinCamera(root);
        if(root.val==0){
            root.val=1;
            count++;
        }
        return count;
    }
    //返回值1代表的是摄像头节点，2代表的是被摄像头检测到的节点
    private int checkMinCamera(TreeNode node) {
        //如果左右子节点均为null，返回0
        if(node.left==null&&node.right==null){
            return 0;
        }
        //如果返回值为0，则当前节点设置为摄像头；返回值为1，则当前节点会被监视到，节点值被设置为2；返回值为2，什么也不干
        if(node.left!=null){
            int treeVal = checkMinCamera(node.left);
            if(treeVal==0){
                count++;
                node.val=1;
                node.left.val=2;
            }else if(treeVal==1){
                node.val=2;
            }
        }
        if(node.right!=null){
            int treeVal = checkMinCamera(node.right);
            //还要判断node是否已经在左遍历中被设值为1
            if(treeVal==0&&node.val!=1){
                count++;
                node.val=1;
                node.right.val=2;
                //此处也要判断是否node已经是1了，不然会把一个监控节点强行改为2，会出错
            }else if(treeVal==1&&node.val!=1){
                node.val=2;
            }
        }
        if(node.val==2){
            return 2;
        }else if(node.val==1){
            return 1;
        }else {
            return 0;
        }
    }
}
```

最优办法就是从树的根部往上走：

- 根节点的父节点一定是一个监控节点（监控节点被标记为1）
- 被监控的节点会被标记为2





# Dynamic Programming

## ----背包问题

![416.分割等和子集1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131373137313330373430372e706e67.png)

## ----01背包问题

有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

在下面的讲解中，我举一个例子：

背包最大重量为4。

物品为：

|       | 重量 | 价值 |
| ----- | ---- | ---- |
| 物品0 | 1    | 15   |
| 物品1 | 3    | 20   |
| 物品2 | 4    | 30   |

问背包能背的物品最大价值是多少？

以下讲解和图示中出现的数字都是以这个例子为例。

### 二维dp数组01背包

1. 确定dp数组以及下标的含义

   对于背包问题，有一种写法， 是使用二维数组，即`dp[i][j]` 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。

   只看这个二维数组的定义，大家一定会有点懵，看下面这个图：

   ![动态规划-背包问题1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131303130333030333336312e706e67.png)

2. 确定递推公式

   再回顾一下dp[i][j]的含义：从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。

   那么可以有两个方向推出来`dp[i][j]`，

   - **不放物品i**：由`dp[i - 1][j]`推出，即背包容量为j，里面不放物品i的最大价值，此时`dp[i][j]`就是`dp[i - 1][j]`。(其实就是当物品i的重量大于背包j的重量时，物品i无法放进背包中，所以被背包内的价值依然和前面相同。)
   - **放物品i**：由`dp[i - 1][j - weight[i]]`推出，`dp[i - 1][j - weight[i]]` 为背包容量为j - weight[i]的时候不放物品i的最大价值，那么`dp[i - 1][j - weight[i]] + value[i]` （物品i的价值），就是背包放物品i得到的最大价值

   所以递归公式： `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])`;

3. dp数组如何初始化

   **关于初始化，一定要和dp数组的定义吻合，否则到递推公式的时候就会越来越乱**。

   首先从`dp[i][j]`的定义出发，如果背包容量j为0的话，即`dp[i][0]`，无论是选取哪些物品，背包价值总和一定为0。如图：

   ![动态规划-背包问题2](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f323032313031313031303330343139322e706e67.png)

   在看其他情况。

   状态转移方程 `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])`; 可以看出i 是由 i-1 推导出来，那么i为0的时候就一定要初始化。

   `dp[0][j]`，即：i为0，存放编号0的物品的时候，各个容量的背包所能存放的最大价值。

   那么很明显当 j < weight[0]的时候，`dp[0][j]` 应该是 0，因为背包容量比编号0的物品重量还小。

   当j >= weight[0]时，`dp[0][j]` 应该是value[0]，因为背包容量放足够放编号0物品。

   此时dp数组初始化情况如图所示：

   ![动态规划-背包问题7](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131303130333130393134302e706e67.png)

   `dp[0][j]` 和 `dp[i][0]` 都已经初始化了，那么其他下标应该初始化多少呢？

   其实从递归公式： `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])`; 可以看出`dp[i][j]` 是由左上方数值推导出来了，那么 其他下标初始为什么数值都可以，因为都会被覆盖。

   **初始-1，初始-2，初始100，都可以！**

   但只不过一开始就统一把dp数组统一初始为0，更方便一些。

   如图：

   ![动态规划-背包问题10](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f2545352538412541382545362538302538312545382541372538342545352538382539322d25453825383325384325453525384325383525453925393725414525453925413225393831302e6a7067.png)

4. 确定遍历顺序

   ![动态规划-背包问题3](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f323032313031313031303331343035352e706e67.png)

   **要理解递归的本质和递推的方向**。

   `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])`; 递归公式中可以看出`dp[i][j]`是靠`dp[i-1][j]`和`dp[i - 1][j - weight[i]]`推导出来的。

   `dp[i-1][j]`和`dp[i - 1][j - weight[i]]` 都在`dp[i][j]`的左上角方向（包括正上方向），那么先遍历物品，再遍历背包的过程如图所示：

   ![动态规划-背包问题5](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f3230323130313130313033323132342e706e67.png)

5. 举例推导dp数组

   ![动态规划-背包问题4](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131383136333432353132392e6a7067.png)

   最终结果就是`dp[2][4]`。

   建议大家此时自己在纸上推导一遍，看看dp数组里每一个数值是不是这样的。

   **做动态规划的题目，最好的过程就是自己在纸上举一个例子把对应的dp数组的数值推导一下，然后在动手写代码！**

```java
//01背包问题
public class Bag01 {
    public static void main(String[] args) {
        int[] weight = {1, 3, 4};
        int[] value = {15, 20, 30};
        int bagsize = 4;
        testweightbagproblem(weight, value, bagsize);
    }

    public static void testweightbagproblem(int[] weight, int[] value, int bagsize){
        int wlen = weight.length, value0 = 0;
        //定义dp数组：dp[i][j]表示背包容量为j时，前i个物品能获得的最大价值
        int[][] dp = new int[wlen + 1][bagsize + 1];
        //初始化：背包容量为0时，能获得的价值都为0
        for (int i = 0; i <= wlen; i++){
            dp[i][0] = value0;
        }
        //遍历顺序：先遍历物品，再遍历背包容量
        //dp的容量为什么是[weight.length+1][bagsize+1]，这个+1都是用来初始化递推公式用的
        for (int i = 1; i <= wlen; i++){
            for (int j = 1; j <= bagsize; j++){
                if (j < weight[i - 1]){
                    dp[i][j] = dp[i - 1][j];
                }else{
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value[i - 1]);
                }
            }
        }
        //打印dp数组
        for (int i = 0; i <= wlen; i++){
            for (int j = 0; j <= bagsize; j++){
                System.out.print(dp[i][j] + " ");
            }
            System.out.print("\n");
        }
    }
}
```



### 滚动数组01背包

对于背包问题其实状态都是可以压缩的。

在使用二维数组的时候，递推公式：`dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])`;

**其实可以发现如果把dp[i - 1]那一层拷贝到dp[i]上，表达式完全可以是：`dp[i][j] = max(dp[i][j], dp[i][j - weight[i]] + value[i])`;**

**与其把dp[i - 1]这一层拷贝到dp[i]上，不如只用一个一维数组了**，只用dp[j]（一维数组，也可以理解是一个滚动数组）。

这就是滚动数组的由来，需要满足的条件是上一层可以重复利用，直接拷贝到当前层。

读到这里估计大家都忘了 dp[i][j]里的i和j表达的是什么了，i是物品，j是背包容量。

**`dp[i][j]` 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少**。

1. 确定dp数组的定义

   在一堆dp数组中，`dp[j]`表示：容量为j的背包，所背的物品价值可以最大为`dp[j]`

2. 一维dp数组的递推公式

   `dp[j]`为 容量为j的背包所背的最大价值，那么如何推导dp[j]呢？

   `dp[j]`可以通过`dp[j - weight[i]]`推导出来，`dp[j - weight[i]]`表示容量为`j - weight[i]`的背包所背的最大价值。

   `dp[j - weight[i]] + value[i]` 表示 容量为 `j - 物品i重量` 的背包 加上 物品i的价值。（也就是容量为j的背包，放入物品i了之后的价值即：`dp[j]`）

   此时`dp[j]`有两个选择，一个是取自己`dp[j]` 相当于 二维dp数组中的`dp[i-1][j]`，即不放物品i，一个是取`dp[j - weight[i]] + value[i]`，即放物品i，指定是取最大的，毕竟是求最大价值

3. 一维dp数组如何初始化

   **关于初始化，一定要和dp数组的定义吻合，否则到递推公式的时候就会越来越乱**。

   dp[j]表示：容量为j的背包，所背的物品价值可以最大为dp[j]，那么dp[0]就应该是0，因为背包容量为0所背的物品的最大价值就是0。

   那么dp数组除了下标0的位置，初始为0，其他下标应该初始化多少呢？

   看一下递归公式：`dp[j] = max(dp[j], dp[j - weight[i]] + value[i])`;

   dp数组在推导的时候一定是取价值最大的数，如果题目给的价值都是正整数那么非0下标都初始化为0就可以了。

   **这样才能让dp数组在递归公式的过程中取的最大的价值，而不是被初始值覆盖了**。

   那么我假设物品价值都是大于0的，所以dp数组初始化的时候，都初始为0就可以了。

4. 一维dp数组遍历顺序

   ```java
   for(int i = 0; i < weight.size(); i++) { // 遍历物品
       for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
           dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
   
       }
   }
   ```

   **这里大家发现和二维dp的写法中，遍历背包的顺序是不一样的！**

   二维dp遍历的时候，背包容量是从小到大，而一维dp遍历的时候，背包是从大到小。

   为什么呢？

   **倒序遍历是为了保证物品i只被放入一次！**。但如果一旦正序遍历了，那么物品0就会被重复加入多次！

   举一个例子：物品0的重量weight[0] = 1，价值value[0] = 15

   如果正序遍历

   dp[1] = dp[1 - weight[0]] + value[0] = 15

   dp[2] = dp[2 - weight[0]] + value[0] = 30

   此时dp[2]就已经是30了，意味着物品0，被放入了两次，所以不能正序遍历。

   为什么倒序遍历，就可以保证物品只放入一次呢？

   倒序就是先算dp[2]

   dp[2] = dp[2 - weight[0]] + value[0] = 15 （dp数组已经都初始化为0）

   dp[1] = dp[1 - weight[0]] + value[0] = 15

   所以从后往前循环，每次取得状态不会和之前取得状态重合，这样每种物品就只取一次了。

   **那么问题又来了，为什么二维dp数组历的时候不用倒序呢？**

   因为对于二维dp，dp[i][j]都是通过上一层即dp[i - 1][j]计算而来，本层的dp[i][j]并不会被覆盖！

   （如何这里读不懂，大家就要动手试一试了，空想还是不靠谱的，实践出真知！）

   **再来看看两个嵌套for循环的顺序，代码中是先遍历物品嵌套遍历背包容量，那可不可以先遍历背包容量嵌套遍历物品呢？**

   不可以！

   因为一维dp的写法，背包容量一定是要倒序遍历（原因上面已经讲了），如果遍历背包容量放在上一层，那么每个dp[j]就只会放入一个物品，即：背包里只放入了一个物品。

   （这里如果读不懂，就在回想一下dp[j]的定义，或者就把两个for循环顺序颠倒一下试试！）

   **所以一维dp数组的背包在遍历顺序上和二维其实是有很大差异的！**，这一点大家一定要注意。

5. 举例推导dp数组

   一维dp，分别用物品0，物品1，物品2 来遍历背包，最终得到结果如下：

   ![动态规划-背包问题9](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303131303130333631343736392e706e67.png)

```java
public static void main(String[] args) {
        int[] weight = {1, 3, 4};
        int[] value = {15, 20, 30};
        int bagWight = 4;
        testWeightBagProblem(weight, value, bagWight);
    }

    public static void testWeightBagProblem(int[] weight, int[] value, int bagWeight){
        int wLen = weight.length;
        //定义dp数组：dp[j]表示背包容量为j时，能获得的最大价值
        int[] dp = new int[bagWeight + 1];
        //遍历顺序：先遍历物品，再遍历背包容量
        for (int i = 0; i < wLen; i++){
            for (int j = bagWeight; j >= weight[i]; j--){
                dp[j] = Math.max(dp[j], dp[j - weight[i]] + value[i]);
            }
        }
        //打印dp数组
        for (int j = 0; j <= bagWeight; j++){
            System.out.print(dp[j] + " ");
        }
    }
```





## [0416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```java
//0416. 分割等和子集
public class Solution8 {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum+=num;
        }
        if(sum%2!=0){
            return false;
        }
        return checkPartition(nums, sum);
    }

    private boolean checkPartition(int[] nums, int sum) {
        int[] dp = new int[sum/2+1];
        for(int i=0; i<nums.length; i++){
            for(int j=dp.length-1; j>=nums[i]; j--){
                dp[j]=Math.max(dp[j],dp[j-nums[i]]+nums[i]);
                if(j==dp.length-1&&dp[j]==j){
                    return true;
                }
            }
        }
        return false;
    }
}
```

本题要求是可以将数组分隔成两个子集，使得两个子集元素和相等。

因为是和相等，所以解决的思路就是判断其中动态生成的一个子集和是否能等于sum/2（这里也是可以直接在最开始的时候直接判断数组元素和是否是一个偶数，如果是奇数可以直接返回false）。

**只有确定了如下四点，才能把01背包问题套到本题上来。**

- 背包的体积为sum / 2
- 背包要放入的商品（集合里的元素）重量为 元素的数值，价值也为元素的数值
- 背包如果正好装满，说明找到了总和为 sum / 2 的子集。
- 背包中每一个元素是不可重复放入。

以上分析完，我们就可以套用01背包，来解决这个问题了。

动规五部曲分析如下：

1. 确定dp数组以及下标的含义

01背包中，dp[j] 表示： 容量为j的背包，所背的物品价值可以最大为dp[j]。

**套到本题，dp[j]表示 背包总容量是j，最大可以凑成j的子集总和为dp[j]**。

1. 确定递推公式

01背包的递推公式为：`dp[j] = max(dp[j], dp[j - weight[i]] + value[i])`;

本题，相当于背包里放入数值，那么物品i的重量是nums[i]，其价值也是nums[i]。

所以递推公式：`dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])`;

1. dp数组如何初始化

在01背包，一维dp如何初始化，已经讲过，

从dp[j]的定义来看，首先dp[0]一定是0。

如果如果题目给的价值都是正整数那么非0下标都初始化为0就可以了，如果题目给的价值有负数，那么非0下标就要初始化为负无穷。

**这样才能让dp数组在递归公式的过程中取的最大的价值，而不是被初始值覆盖了**。

本题题目中 只包含正整数的非空数组，所以非0下标的元素初始化为0就可以了。





## [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

```java
//1049. 最后一块石头的重量 II
public class Solution9 {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int stone : stones) {
            sum += stone;
        }
        int[] dp = new int[sum/2+1];
        for(int i=0; i<stones.length; i++){
            for(int j=dp.length-1; j>=stones[i]; j--){
                dp[j] = Math.max(dp[i], dp[j-stones[i]]+stones[i]);
            }
        }
        //dp数组最后的元素就是重量为sum/2的背包可以携带的最大重量，即把数组中一份石头拆分成了两份石头，比较两份石头大小，输出他们差的绝对值
        return sum-2*dp[dp.length-1]>0 ? sum-2*dp[dp.length-1] : 2*dp[dp.length-1]-sum;
    }
}
```

本题其实就是尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小，**这样就化解成01背包问题了**。

是不是感觉和昨天讲解的[416. 分割等和子集](https://programmercarl.com/0416.分割等和子集.html)非常像了。

本题物品的重量为store[i]，物品的价值也为store[i]。

对应着01背包里的物品重量weight[i]和 物品价值value[i]。







## 0416和1049的总结

在学习了01背包后，经历了这两道题目，始终就是要想清楚，dp数组的i和j分别代表什么，要把不同属性对应到相应01背包中。例如i其实很好定义，一般都是某些实例的物体，比如石头，那i就代表了不同的石头；在这几题中，j始终代表的是背包的重量，这时候就要思考一下到底什么是“背包”？根据题意决定背包是一个非常重要的过程。





## [0494. 目标和](https://leetcode-cn.com/problems/target-sum/)（重要，第一次涉及组合排列问题）

```java
//0494.目标和
public class Solution10 {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        //题目已经给出所有元素都大于0，所以sum不用带abs
        if((target + sum) % 2 == 1 || Math.abs(target) > sum){
            return 0;
        }
        //加分总和为x，那么减法总和就为sum-x，那么就有如下公式：x-(sum-x)=target -> x=(target+sum)/2
        int addition = (target + sum) / 2;
        if (addition < 0){
            addition = -addition;
        }
        int[] dp = new int[addition+1];
        dp[0] = 1;
        for (int i = 0; i < nums.length; i++){
            for(int j = dp.length-1; j >= nums[i]; j--){
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[dp.length-1];
    }
}
```

首先想到的是把数组元素拆分成两部分，一部分是加法的部分，另一部分是减法的部分，这样两部分相加如果等于target，那么说明满足题目要求。

本题要如何使表达式结果为target，既然为target，那么就一定有 left组合 - right组合 = target。left + right等于sum，而sum是固定的。

公式来了， left - (sum - left) = target -> left = (target + sum)/2 。

==**这次和之前遇到的背包问题不一样了，之前都是求容量为j的背包，最多能装多少。**==

==**本题则是装满有几种方法。其实这就是一个组合问题了。**==

- 可以先筛选部分结果，因为left肯定得是一个整数，如果不为整数则直接返回0。

- 递推公式：

  填满容量为j - nums[i]的背包，有dp[j - nums[i]]种方法。

  那么只要搞到nums[i]的话，凑成dp[j]就有dp[j - nums[i]] 种方法。

  例如：dp[j]，j 为5，

  - 已经有一个1（nums[i]） 的话，有 dp[4]种方法 凑成 dp[5]。
  - 已经有一个2（nums[i]） 的话，有 dp[3]种方法 凑成 dp[5]。
  - 已经有一个3（nums[i]） 的话，有 dp[2]中方法 凑成 dp[5]
  - 已经有一个4（nums[i]） 的话，有 dp[1]中方法 凑成 dp[5]
  - 已经有一个5 （nums[i]）的话，有 dp[0]中方法 凑成 dp[5]

  那么凑整dp[5]有多少方法呢，也就是把 所有的 dp[j - nums[i]] 累加起来。

  所以求组合类问题的公式，都是类似这种：

  ```java
  dp[j] += dp[j - nums[i]]
  ```

- 从递归公式可以看出，在初始化的时候dp[0] 一定要初始化为1，因为dp[0]是在公式中一切递推结果的起源，如果dp[0]是0的话，递归结果将都是0。

  dp[0] = 1，理论上也很好解释，装满容量为0的背包，有1种方法，就是装0件物品。

  dp[j]其他下标对应的数值应该初始化为0，从递归公式也可以看出，dp[j]要保证是0的初始值，才能正确的由dp[j - nums[i]]推导出来。







## [0474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

```java
//0474.一和零
public class Solution11 {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m+1][n+1];
        //这个i的遍历依旧和之前一样，是对strs中各个元素的遍历
        for (int i = 0; i < strs.length; i++){
            int mPart = 0;//0
            int nPart = 0;//1
            for (int j = 0; j < strs[i].length(); j++){
                if (strs[i].charAt(j) == '0'){
                    mPart++;
                }else {
                    nPart++;
                }
            }
            //这里的背包是一个二维背包，需要好好理解一下，这个背包的大小有两层关系，但依旧运用的是滚动数组。
            for (int j = m; j >= mPart; j--){
                for (int k = n; k >= nPart; k--){
                    dp[j][k] = Math.max(dp[j][k], dp[j - mPart][k - nPart] + 1);
                }
            }
        }
        return dp[m][n];
    }
}
```

**本题中strs 数组里的元素就是物品，每个物品都是一个！**

**而m 和 n相当于是一个背包，两个维度的背包**。

理解成多重背包的同学主要是把m和n混淆为物品了，感觉这是不同数量的物品，所以以为是多重背包。

但本题其实是01背包问题！

这不过这个背包有两个维度，一个是m 一个是n，而不同长度的字符串就是不同大小的待装物品。

开始动规五部曲：

1. 确定dp数组（dp table）以及下标的含义

**`dp[i][j]`：最多有i个0和j个1的strs的最大子集的大小为`dp[i][j]`**。

1. 确定递推公式

dp[i][j] 可以由前一个strs里的字符串推导出来，strs里的字符串有zeroNum个0，oneNum个1。

dp[i][j] 就可以是 `dp[i - zeroNum][j - oneNum] + 1`。

然后我们在遍历的过程中，取dp[i][j]的最大值。

所以递推公式：`dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)`;

此时大家可以回想一下01背包的递推公式：`dp[j] = max(dp[j], dp[j - weight[i]] + value[i])`;

对比一下就会发现，字符串的zeroNum和oneNum相当于物品的重量（weight[i]），字符串本身的个数相当于物品的价值（value[i]）。

**这就是一个典型的01背包！** 只不过物品的重量有了两个维度而已。





## ----完全背包

有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品都有无限个（也就是可以放入背包多次）**，求解将哪些物品装入背包里物品价值总和最大。

**完全背包和01背包问题唯一不同的地方就是，每种物品有无限件**。

首先在回顾一下01背包的核心代码：

```java
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}
```

我们知道01背包内嵌的循环是从大到小遍历，为了保证每个物品仅被添加一次。

而完全背包的物品是可以添加多次的，所以要从小到大去遍历，即：

```java
// 先遍历物品，再遍历背包
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = weight[i]; j <= bagWeight ; j++) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

    }
}
```

dp状态图如下：

![动态规划-完全背包](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303132363130343531303130362e6a7067.jpeg)



### 求组合数

两个for循环的先后顺序可就有说法了。

我们先来看 外层for循环遍历物品（钱币），内层for遍历背包（金钱总额）的情况。

```java
for (int i = 0; i < coins.size(); i++) { // 遍历物品
    for (int j = coins[i]; j <= amount; j++) { // 遍历背包容量
        dp[j] += dp[j - coins[i]];
    }
}
```

假设：coins[0] = 1，coins[1] = 5。

那么就是先把1加入计算，然后再把5加入计算，得到的方法数量只有{1, 5}这种情况。而不会出现{5, 1}的情况。

**所以这种遍历顺序中dp[j]里计算的是组合数！**



### 求排列数

```java
for (int j = 0; j <= amount; j++) { // 遍历背包容量
    for (int i = 0; i < coins.size(); i++) { // 遍历物品
        if (j - coins[i] >= 0) dp[j] += dp[j - coins[i]];
    }
}
```

背包容量的每一个值，都是经过 1 和 5 的计算，包含了{1, 5} 和 {5, 1}两种情况。

**此时dp[j]里算出来的就是排列数！**





## [0518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

```java
//0518. 零钱兑换 II
public class Solution12 {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount+1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++){
            for (int j = coins[i]; j <= amount; j++){
                //求组合数的公式
                dp[j] += dp[j-coins[i]];
            }
        }
        return dp[dp.length-1];
    }
}
```

这是一道完全背包的题目，然后解题思路是要用求组合数的公式:`dp[j] += dp[j-coins[i]]`。











## [0377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

```java
//0377. 组合总和 Ⅳ
public class Solution13 {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        dp[0] = 1;
        //先循环遍历背包，再循环遍历物品，就可以用来求排列数！
        for (int j = 0; j <= target; j++){
            for (int i = 0; i < nums.length; i++){
                if (j >= nums[i]){
                    dp[j] += dp[j - nums[i]];
                }
            }
        }
        return dp[dp.length-1];
    }
}
```

完全背包，求排列数，具体区别看上面完全背包的解析。





## 老题重做-求排列数0070.爬楼梯

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**这就是一道完全背包的求排列数问题。**

```java
//0070. 爬楼梯
public class Solution2 {
    //递归，超时
    private int count=0;
    public int climbStairs(int n) {
        checkStairs(n,0);
        return count;
    }
    private void checkStairs(int n, int sum) {
        if(sum>=n){
            if(sum==n){
                count++;
            }
            return;
        }
        checkStairs(n, sum+1);
        checkStairs(n, sum+2);
    }
    //动态规划
    public int climbStairs2(int n) {
        if(n<3){
            return n;
        }
        int[] dp = new int[n+1];
        dp[0]=0;
        dp[1]=1;
        dp[2]=2;
        for(int index=3; index<=n; index++){
            dp[index] = dp[index-1] + dp[index-2];
        }
        return dp[n];
    }
    //动态规划-完全背包
    public int climbStair3(int n) {
        int[] dp = new int[n+1];
        for (int j = 0; j <= n; j++){
            for (int i = 0; i < 2; i++){
                if (j >= i + 1){
                    dp[j] = dp[j - (i + 1)];
                }
            }
        }
        return dp[dp.length-1];
    }
}
```







## [0322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```java
//0322.零钱兑换
public class Solution14 {
    public int coinChange(int[] coins, int amount) {
        int max = Integer.MAX_VALUE;
        int[] dp = new int[amount + 1];
        for (int j = 0; j < dp.length; j++) {
            dp[j] = max;
        }
        dp[0] = 0;
        for (int i = 0; i < coins.length ; i++){
            for (int j = coins[i]; j <= amount; j++){
                if(dp[j - coins[i]] != max){
                    dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1);
                }
            }
        }
        return dp[amount] == max ? -1 : dp[amount];
    }
}
```

> 在本题中，最开始我对dp数组的赋值是用的for each增强循环遍历，发现并不能把值成功赋予其中元素

```java
		for (int i : dp) {
            i = Integer.MAX_VALUE;//这样并不能给dp数组中的元素赋值
        }
```



**思路**

1. 确定dp数组以及下标的含义

   **dp[j]：凑足总额为j所需钱币的最少个数为dp[j]**

2. 确定递推公式

   得到dp[j]（考虑coins[i]），只有一个来源，dp[j - coins[i]]（没有考虑coins[i]）。

   凑足总额为j - coins[i]的最少个数为dp[j - coins[i]]，那么只需要加上一个钱币coins[i]即dp[j - coins[i]] + 1就是dp[j]（考虑coins[i]）

   所以dp[j] 要取所有 dp[j - coins[i]] + 1 中最小的。

   递推公式：dp[j] = min(dp[j - coins[i]] + 1, dp[j]);

3. dp数组如何初始化

   首先凑足总金额为0所需钱币的个数一定是0，那么dp[0] = 0;

   其他下标对应的数值呢？

   考虑到递推公式的特性，dp[j]必须初始化为一个最大的数，否则就会在min(dp[j - coins[i]] + 1, dp[j])比较的过程中被初始值覆盖。

   所以下标非0的元素都是应该是最大值。





## [0279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```java
//0279.完全平方数
public class Solution15 {
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        int max = Integer.MAX_VALUE;
        for (int i = 0; i <= n; i++){
            dp[i] = max;
        }
        dp[0] = 0;
        //i * i <= n实际上就是变相求出了物品的数量
        for (int i = 1; i * i <= n; i++){
            for (int j = i * i; j <= n; j++){
                if(dp[j - i * i] != max){
                    dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
                }
            }
        }
        return dp[n] == max ? 0 : dp[n];
    }
}
```

**我来把题目翻译一下：完全平方数就是物品（可以无限件使用），凑个正整数n就是背包，问凑满这个背包最少有多少物品？**

感受出来了没，这么浓厚的完全背包氛围，而且和昨天的题目[动态规划：322. 零钱兑换](https://programmercarl.com/0322.零钱兑换.html)就是一样一样的！

动规五部曲分析如下：

1. 确定dp数组（dp table）以及下标的含义

   **dp[j]：和为j的完全平方数的最少数量为dp[j]**

2. 确定递推公式

   dp[j] 可以由`dp[j - i * i]`推出， `dp[j - i * i] + 1` 便可以凑成dp[j]。

   此时我们要选择最小的dp[j]，所以递推公式：`dp[j] = min(dp[j - i * i] + 1, dp[j])`;

3. dp数组如何初始化

   dp[0]表示 和为0的完全平方数的最小数量，那么dp[0]一定是0。

   有同学问题，那0 * 0 也算是一种啊，为啥dp[0] 就是 0呢？

   看题目描述，找到若干个完全平方数（比如 1, 4, 9, 16, ...），题目描述中可没说要从0开始，dp[0]=0完全是为了递推公式。

   非0下标的dp[j]应该是多少呢？

   从递归公式`dp[j] = min(dp[j - i * i] + 1, dp[j])`;中可以看出每次dp[j]都要选最小的，**所以非0下标的dp[j]一定要初始为最大值，这样dp[j]在递推的时候才不会被初始值覆盖**。







## [0139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

```java
//0139.单词拆分
public class Solution16 {
    public boolean wordBreak(String s, List<String> wordDict) {
        Boolean[] dp = new Boolean[s.length()+1];
        for (int i = 0; i <=s.length(); i++){
            dp[i] = false;
        }
        dp[0] = true;//用于推导，题目中不会出现空字符串
        for (int j = 0; j <= dp.length - 1; j++){
            for (int i = 0; i <wordDict.size(); i++){
                if (j >= wordDict.get(i).length()){
                    for (String s1 : wordDict) {
                        //因为dp[0]仅是一个占位符，用来推导，所以dp[]背包的实际下标是从1开始，在substring中(i,j)j就没必要+1了。
                        if(s.substring(j - wordDict.get(i).length(), j).equals(s1) && dp[j - wordDict.get(i).length()] == true){
                            dp[j] = true;
                        }
                    }
                }
            }
        }
        return dp[dp.length-1];
    }
}
```

**思路**：

1. 确定dp数组以及下标的含义

   **dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词**。

2. 确定递推公式

   如果确定dp[j] 是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。

   所以递推公式是 if([j, i] 这个区间的子串出现在字典里 && dp[j]是true) 那么 dp[i] = true。

3. dp数组如何初始化

   从递归公式中可以看出，dp[i] 的状态依靠 dp[j]是否为true，那么dp[0]就是递归的根基，dp[0]一定要为true，否则递归下去后面都都是false了。

   那么dp[0]有没有意义呢？

   dp[0]表示如果字符串为空的话，说明出现在字典里。

   但题目中说了“给定一个非空字符串 s” 所以测试数据中不会出现i为0的情况，那么dp[0]初始为true完全就是为了推导公式。

   下标非0的dp[i]初始化为false，只要没有被覆盖说明都是不可拆分为一个或多个在字典中出现的单词。

4. 确定遍历顺序

   本题最终要求的是是否都出现过，所以对出现单词集合里的元素是组合还是排列，并不在意！

   **那么本题使用求排列的方式，还是求组合的方式都可以**。





## ----打家劫舍

## [0198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```java
//0198.打家劫舍
public class Solution17 {
    public int rob(int[] nums) {
        if (nums.length == 1){
            return nums[0];
        }
        int[] dp = new int[nums.length+1];
        dp[0] = 0;
        dp[1] = nums[0];
        dp[2] = nums[1];
        for (int i = 3; i <=nums.length; i++){
            dp[i] = Math.max(dp[i - 2], dp[i - 3]) + nums[i-1];
        }
        return dp[dp.length - 1] >= dp[dp.length - 2] ? dp[dp.length - 1] : dp[dp.length - 2];
    }
}
```

**思路**：

因为不能打劫相邻的住户，所以一定是要跳过一个住户去打劫，那么初步判断递推公式为`dp[i] = dp[i - 2] + nums[i - 1]`，这里是nums[i - 1]的原因是dp数组有个空位dp[0]，nums[0]已经对应的是dp[1]了。

dp[i]最终的递推其实有两个分支，一个是由dp[i - 3]推出，所以最终`dp[i] = Math.max(dp[i - 2], dp[i - 3]) + nums[i - 1]`。





## [0213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

```java
//0213.打家劫舍II
public class Solution18 {
    public int rob(int[] nums) {
        if (nums.length == 1){
            return nums[0];
        }
        if (nums.length == 2){
            return nums[0] >= nums[1] ? nums[0] : nums[1];
        }
        //这里不是nums.length+1，因为我们要考虑两种情况，一个是index=0时开始打劫，和index=1时开始打劫，然后把它前面的那一个住户排除数组
        //即从index=0开始遍历的话，那么index=nums.length-1的元素就要从数组中排除；index=1开始遍历的话，那么index=0的元素就要从数组排除
        int[] dp = new int[nums.length];
        dp[0] = 0;
        dp[1] = nums[0];
        dp[2] = nums[1];
        for (int i = 3; i <=nums.length - 1; i++){
            dp[i] = Math.max(dp[i - 2], dp[i - 3]) + nums[i-1];
        }
        int max = dp[dp.length - 1] >= dp[dp.length - 2] ? dp[dp.length - 1] : dp[dp.length - 2];
        dp[0] = 0;
        dp[1] = nums[1];
        dp[2] = nums[2];
        for (int i = 3; i <=nums.length - 1; i++){
            dp[i] = Math.max(dp[i - 2], dp[i - 3]) + nums[i];
        }
        max = Math.max(max, dp[dp.length - 1] >= dp[dp.length - 2] ? dp[dp.length - 1] : dp[dp.length - 2]);
        return max;
    }
}
```

与打家劫舍1不同的地方在于，因为这次的住宅是一个环形，即头尾相连，那么我们就考虑两种情况：

1. 从index=0的数组元素开始遍历，那么index=0住宅前面的那一个住宅就是index=nums.length-1的住宅，这个住宅一定不能被选中，所以直接从dp数组中移除。
2. 从index=1的数组元素开始遍历，index=1住宅前面的住宅是index=0的住宅，把其从dp数组中移除



那么就相当于做两次循环，每次循环的递推公式和打家劫舍1一样：

```java
dp[i] = Math.max(dp[i - 2], dp[i - 3]) + nums[i-1];
```

但是初始化是有所不同的：

1. 从index=0开始遍历，那么初始化还是和之前一样，dp[1]=nums[0]，dp[2]=nums[1]。但是循环终点不是nums.length了，而是nums.length-1了，因为上面已经说过了，nums[nums.length-1]被移除了dp数组。
2. 从index=1开始遍历，初始化就不同了，dp[1]=nums[1]，dp[2]=nums[2]，因为nums[0]被移除了dp数组。



最后会得到两个遍历结果，比较两个之间的最大值就是答案。







## [0337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

```java
//0337. 打家劫舍 III
public class Solution19 {
    //public int nodeAmount = 0;
    public int rob(TreeNode root) {
        //int[] dp = new int[10001];
        int[] res = checkRob(root);
        return Math.max(res[0], res[1]);
    }

    private int[] checkRob(TreeNode root) {
        if (root.left == null && root.right == null){
            int[] arr = {root.val, 0};
            return arr;
        }
        int left[];
        if (root.left != null){
            left = checkRob(root.left);
        }else {
            left = new int[]{0, 0};
        }
        int right[];
        if (root.right != null){
            right = checkRob(root.right);
        }else {
            right = new int[]{0, 0};
        }
        int[] arr = {Math.max(left[0] + right[0], left[1] + right[1] + root.val), left[0] + right[0]};
        return arr;
    }
}
```

**本人思路**：

首先看到注释的两行代码，我一开始对于二叉树得dp数组确实是不知道怎么创建。最后还是打算用递归来传递动态每个节点的最大值返回给上一层，这样最后传回root节点时就是整体最大，用的贪心在做。

1. 利用后续遍历来做这个题目，先看左右子节点，再看本节点这样的顺序。
2. 首先递归函数每次要传回一个大小为2的int数组，int[0]用来存放当前节点的最大值，int[1]用来存放当前节点左+右节点的最大值。
3. 需要这两个参数的意义是，递推公式`Math.max(left[0] + right[0], left[1] + right[1] + root.val`，所以需要知道当前节点子节点的左右节点最大值和。left[0] + right[0]就是当前节点左右两个节点的最大值和，即一旦偷了这两个房子，就肯定偷不了当前节点的房子，left[1] + right[1] + root.val就是偷当前房子，那么其他就必须跳过其左右两个节点的房子从剩余的一路偷过来的。
4. 根据这样的递推，最后返回的arr[0]就是root节点的最大值，即能偷的最大值。





**动态规划**：

**这道题目算是树形dp的入门题目，因为是在树上进行状态转移，我们在讲解二叉树的时候说过递归三部曲，那么下面我以递归三部曲为框架，其中融合动规五部曲的内容来进行讲解**。

1. 确定递归函数的参数和返回值

   这里我们要求一个节点 偷与不偷的两个状态所得到的金钱，那么返回值就是一个长度为2的数组。

   参数为当前节点，代码如下：

   ```c++
   vector<int> robTree(TreeNode* cur) {
   ```

   其实这里的返回数组就是dp数组。

   所以dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱。

   **所以本题dp数组就是一个长度为2的数组！**

   那么有同学可能疑惑，长度为2的数组怎么标记树中每个节点的状态呢？

   **别忘了在递归的过程中，系统栈会保存每一层递归的参数**。

   如果还不理解的话，就接着往下看，看到代码就理解了哈。

2. 确定终止条件

   在遍历的过程中，如果遇到空节点的话，很明显，无论偷还是不偷都是0，所以就返回

   ```c++
   if (cur == NULL) return vector<int>{0, 0};
   ```

   这也相当于dp数组的初始化

3. 确定遍历顺序

   首先明确的是使用后序遍历。 因为通过递归函数的返回值来做下一步计算。

   通过递归左节点，得到左节点偷与不偷的金钱。

   通过递归右节点，得到右节点偷与不偷的金钱。

   代码如下：

   ```c++
   // 下标0：不偷，下标1：偷
   vector<int> left = robTree(cur->left); // 左
   vector<int> right = robTree(cur->right); // 右
   // 中
   ```

4. 确定单层递归的逻辑

   如果是偷当前节点，那么左右孩子就不能偷，val1 = cur->val + left[0] + right[0]; （**如果对下标含义不理解就在回顾一下dp数组的含义**）

   如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的，所以：val2 = max(left[0], left[1]) + max(right[0], right[1]);

   最后当前节点的状态就是{val2, val1}; 即：{不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}

   代码如下：

   ```c++
   vector<int> left = robTree(cur->left); // 左
   vector<int> right = robTree(cur->right); // 右
   
   // 偷cur
   int val1 = cur->val + left[0] + right[0];
   // 不偷cur
   int val2 = max(left[0], left[1]) + max(right[0], right[1]);
   return {val2, val1};
   ```

5. 举例推导dp数组

   ![337.打家劫舍III](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/68747470733a2f2f636f64652d7468696e6b696e672e63646e2e626365626f732e636f6d2f706963732f3333372e2545362538392539332545352541452542362545352538412541422545382538382538444949492e6a7067.png)



```java
public int rob3(TreeNode root) {
        int[] res = robAction1(root);
        return Math.max(res[0], res[1]);
    }

    int[] robAction1(TreeNode root) {
        int res[] = new int[2];
        if (root == null)
            return res;

        int[] left = robAction1(root.left);
        int[] right = robAction1(root.right);

        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];
        return res;
    }
```



==**结果来看就是实际上自己写的就是这种思想了。**==







## ----买卖股票问题

## [0121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

思路：

1. 确定dp数组（dp table）以及下标的含义

   dp[i][0] 表示第i天持有股票所得最多现金 ，**这里可能有同学疑惑，本题中只能买卖一次，持有股票之后哪还有现金呢？**

   其实一开始现金是0，那么加入第i天买入股票现金就是 -prices[i]， 这是一个负数。

   dp[i][1] 表示第i天不持有股票所得最多现金

   **注意这里说的是“持有”，“持有”不代表就是当天“买入”！也有可能是昨天就买入了，今天保持持有的状态**

   很多同学把“持有”和“买入”没分区分清楚。

   在下面递推公式分析中，我会进一步讲解。

2. 确定递推公式

   如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来

   - 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：`dp[i - 1][0]`
   - 第i天买入股票，所得现金就是买入今天的股票后所得现金即：-prices[i]

   那么dp[i][0]应该选所得现金最大的，所以`dp[i][0] = max(dp[i - 1][0], -prices[i])`;

   如果第i天不持有股票即dp[i][1]， 也可以由两个状态推出来

   - 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：`dp[i - 1][1]`
   - 第i天卖出股票，所得现金就是按照今天股票佳价格卖出后所得现金即：`prices[i] + dp[i - 1][0]`

   同样dp[i][1]取最大的，`dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0])`;

3. dp数组如何初始化

   由递推公式 `dp[i][0] = max(dp[i - 1][0], -prices[i])`; 和 `dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0])`;可以看出

   其基础都是要从`dp[0][0]`和`dp[0][1]`推导出来。

   那么`dp[0][0]`表示第0天持有股票，此时的持有股票就一定是买入股票了，因为不可能有前一天推出来，所以`dp[0][0] -= prices[0]`;

   `dp[0][1]`表示第0天不持有股票，不持有股票那么现金就是0，所以`dp[0][1] = 0`;

```java
//0121.买卖股票的最佳时机
public class Solution20 {
    public int maxProfit(int[] prices){
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];//第一天持有股票时所得最多现金
        dp[0][1] = 0;//第一天不持有股票时所得最多现金
        for (int i =1; i < dp.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[dp.length-1][1];
    }
}
```



动态数组版本：

```java
  public int maxProfit(int[] prices) {
    int[] dp = new int[2];
    // 记录一次交易，一次交易有买入卖出两种状态
    // 0代表持有，1代表卖出
    dp[0] = -prices[0];
    dp[1] = 0;
    // 可以参考斐波那契问题的优化方式
    // 我们从 i=1 开始遍历数组，一共有 prices.length 天，
    // 所以是 i<=prices.length
    for (int i = 1; i <= prices.length; i++) {
      // 前一天持有；或当天买入
      dp[0] = Math.max(dp[0], -prices[i - 1]);
      // 如果 dp[0] 被更新，那么 dp[1] 肯定会被更新为正数的 dp[1]
      // 而不是 dp[0]+prices[i-1]==0 的0，
      // 所以这里使用会改变的dp[0]也是可以的
      // 当然 dp[1] 初始值为 0 ，被更新成 0 也没影响
      // 前一天卖出；或当天卖出, 当天要卖出，得前一天持有才行
      dp[1] = Math.max(dp[1], dp[0] + prices[i - 1]);
    }
    return dp[1];
  }
```





## [0122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```java
//0122. 买卖股票的最佳时机 II
public class Solution21 {
    public int maxProfit(int[] prices) {
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];//第一天持有股票时所得最多现金
        dp[0][1] = 0;//第一天不持有股票时所得最多现金
        for (int i =1; i < dp.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[dp.length-1][1];
    }
}
```

代码和0121.买卖股票的最佳时机基本一样，逻辑不同是0122可以多次买入卖出股票，但是0121不行，所以在`dp[i][0]`的递推公式上要做出一些改动，即`dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i])` 。

但是和0121一样，本题动态规划的时间都比贪心要花费更多时间。





## [0123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

```java
//0123. 买卖股票的最佳时机 III
public class Solution22 {
    public int maxProfit(int[] prices) {
        if (prices.length == 1){
            return 0;
        }
        int[] dp2 = new int[prices.length];
        dp2[0] = 0;
        for (int i = 0; i < dp2.length; i++){
            int[][] dp = new int[i + 1][2];
            dp[0][0] = -prices[0];
            dp[0][1] = 0;
            for (int k = 1; k < dp.length; i++){
                dp[k][0] = Math.max(dp[k - 1][0], -prices[k]);
                dp[k][1] = Math.max(dp[k - 1][1], dp[k - 1][0] + prices[k]);
            }
            int[][] dpa = new int[dp2.length - 1 - i][2];
            if (dpa.length > 0){
                dpa[0][0] = -prices[i + 1];
                dpa[0][1] = 0;
            }
            for (int j = 1; j < dpa.length; j++){
                dpa[j][0] = Math.max(dpa[j - 1][0], -prices[i + 1 + j]);
                dpa[j][1] = Math.max(dpa[j - 1][1], dpa[j - 1][0] + prices[i + 1 + j]);
            }
            if (i == 0){
                dp2[i] = dp[dp.length - 1][1] + dpa[dpa.length - 1][1];
            }else if (i == dp2.length - 1){
                dp2[i] = Math.max(dp2[i - 1], dp[dp.length - 1][1]);
            }else {
                dp2[i] = Math.max(dp2[i - 1], dp[dp.length - 1][1] + dpa[dpa.length - 1][1]);
            }
        }
        return dp2[dp2.length - 1];
    }

    public int maxProfit2(int[] prices) {
        int[][] dp = new int[prices.length][5];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[0][2] = 0;
        dp[0][3] = -prices[0];
        dp[0][4] = 0;
        for (int i = 1; i < dp.length; i++){
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }
        return dp[dp.length - 1][4] >= dp[dp.length - 1][2] ? dp[dp.length - 1][4] : dp[dp.length - 1][2];
    }
}
```

写了两种方法，第一种方法是相当于把数组拆分成两部分，分别做0121.买卖股票的最佳时机的操作，就是没法ac，运行超时。

第二种方法是：

1. 确定dp数组以及下标的含义

   一天一共就有五个状态，

   1. 没有操作
   2. 第一次买入
   3. 第一次卖出
   4. 第二次买入
   5. 第二次卖出

   `dp[i][j]`中 i表示第i天，j为 [0 - 4] 五个状态，`dp[i][j]`表示第i天状态j所剩最大现金。

2. 确定递推公式

   需要注意：`dp[i][1]`，**表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票，这是很多同学容易陷入的误区**。

   达到`dp[i][1]`状态，有两个具体操作：

   - 操作一：第i天买入股票了，那么`dp[i][1] = dp[i-1][0] - prices[i]`
   - 操作二：第i天没有操作，而是沿用前一天买入的状态，即：`dp[i][1] = dp[i - 1][1]`

   那么`dp[i][1]`究竟选 `dp[i-1][0] - prices[i]`，还是`dp[i - 1][1]`呢？

   一定是选最大的，所以 `dp[i][1] = max(dp[i-1][0] - prices[i], dp[i - 1][1])`;

   同理`dp[i][2]`也有两个操作：

   - 操作一：第i天卖出股票了，那么`dp[i][2] = dp[i - 1][1] + prices[i]`
   - 操作二：第i天没有操作，沿用前一天卖出股票的状态，即：`dp[i][2] = dp[i - 1][2]`

   所以`dp[i][2] = max(dp[i - 1][1] + prices[i], dp[i - 1][2])`

   同理可推出剩下状态部分：

   `dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])`;

   `dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])`;

3. dp数组如何初始化

   第0天没有操作，这个最容易想到，就是0，即：`dp[0][0] = 0`;

   第0天做第一次买入的操作，`dp[0][1] = -prices[0]`;

   第0天做第一次卖出的操作，这个初始值应该是多少呢？

   首先卖出的操作一定是收获利润，整个股票买卖最差情况也就是没有盈利即全程无操作现金为0，

   从递推公式中可以看出每次是取最大值，那么既然是收获利润如果比0还小了就没有必要收获这个利润了。

   所以`dp[0][2] = 0`;

   第0天第二次买入操作，初始值应该是多少呢？应该不少同学疑惑，第一次还没买入呢，怎么初始化第二次买入呢？

   第二次买入依赖于第一次卖出的状态，其实相当于第0天第一次买入了，第一次卖出了，然后在买入一次（第二次买入），那么现在手头上没有现金，只要买入，现金就做相应的减少。

   所以第二次买入操作，初始化为：`dp[0][3] = -prices[0]`;

   同理第二次卖出初始化`dp[0][4] = 0`;





## [0188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```java
//0188. 买卖股票的最佳时机 IV
public class Solution23 {
    public int maxProfit(int k, int[] prices) {
        if (k == 0 || prices.length == 0){
            return 0;
        }
        int[][] dp = new int[prices.length][1 + 2 * k];
        dp[0][0] = 0;
        for (int i = 1; i < dp[0].length; i++){
            if (i % 2 == 1){
                dp[0][i] = -prices[0];
            }else {
                dp[0][i] = 0;
            }
        }
        for (int i = 1; i < dp.length; i++){
            dp[i][0] = dp[i - 1][0];
            for (int j = 1; j < dp[0].length; j++){
                if (j % 2 == 1){
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i]);
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i]);
                }
            }
        }
        int max = 0;
        for (int i = 2; i < dp[0].length; i++){
            max = Math.max(max, dp[dp.length - 1][i]);
        }
        return max;
    }
}
```

与0123.买卖股票的最佳时机 III是同一个逻辑，对其进行总结归纳：

1. 确定dp数组以及下标的含义

   j的状态表示为：

   - 0 表示不操作
   - 1 第一次买入
   - 2 第一次卖出
   - 3 第二次买入
   - 4 第二次卖出
   - .....

   **除了0以外，偶数就是卖出，奇数就是买入**。

2. 确定递推公式

   - `dp[i][0] = dp[i - 1][0]`，j=0时这个是一直不变的
   - 之后当j>0且为奇数时，是买入的时间点，所以`Math.max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])`
   - 当j>0且为偶数时，是卖出的时间点，所以`dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i])`
   - 最后对`dp[dp.length - 1]`中的所有偶数元素做遍历，选出最大值即可（偶数是卖出，卖出一定比同阶段的奇数元素大）。









## [0309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```java
//0309. 最佳买卖股票时机含冷冻期
public class Solution24 {
    public int maxProfit(int[] prices) {
        int[][] dp = new int[prices.length][4];
        dp[0][0] = -prices[0];//买入股票状态
        dp[0][1] = 0;//不持有股票，已经卖出股票超过一天
        dp[0][2] = 0;//不持有股票，今天刚卖出股票
        dp[0][3] = 0;//冷冻期状态
        for (int i = 1; i < dp.length; i++) {
            dp[i][0] = Math.max(Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]), dp[i - 1][3] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][3]);
            dp[i][2] = dp[i - 1][0] + prices[i];
            dp[i][3] = dp[i - 1][2];
        }
        return Math.max(Math.max(dp[dp.length - 1][1], dp[dp.length - 1][2]), dp[dp.length - 1][3]);
    }
}
```

思路：

1. 确定dp数组以及下标的含义

   `dp[i][j]`，第i天状态为j，所剩的最多现金为`dp[i][j]`。

   **其实本题很多同学搞的比较懵，是因为出现冷冻期之后，状态其实是比较复杂度**，例如今天买入股票、今天卖出股票、今天是冷冻期，都是不能操作股票的。 具体可以区分出如下四个状态：

   - 状态一：买入股票状态（今天买入股票，或者是之前就买入了股票然后没有操作）
   - 卖出股票状态，这里就有两种卖出股票状态
     - 状态二：两天前就卖出了股票，度过了冷冻期，一直没操作，今天保持卖出股票状态
     - 状态三：今天卖出了股票
   - 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！

2. 确定递推公式

   - `dp[i][0]`可以由三个状态推导过来，所以递推公式为`Math.max(Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]), dp[i - 1][3] - prices[i])`，若i-1为刚卖出股票的j=2，在i时状态就是冷冻期，所以不能买入股票。
   - `dp[i][1]`即此时股票已经卖出超过1天，度过冷却期，那么它可以由j=1和j=3推出，因为当i-1状态为冷冻期时，i就已经是状态1即卖出股票超过1天的状态了。递推公式为`Math.max(dp[i - 1][1], dp[i - 1][3])`。
   - `dp[i][2]`状态定义为今天刚卖出股票，所以只能i-1时买入股票的状态推导过来，递推公式为`dp[i - 1][0] + prices[i]`。
   - `dp[i][3]`状态为冷冻期，所以也只能由i-1时刚卖出股票的操作推来，则递推公式为`dp[i - 1][2]`。







## [0714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

```java
//0714.买卖股票的最佳时机含手续费
public class Solution25 {
    public int maxProfit(int[] prices, int fee) {
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];//第一天持有股票时所得最多现金
        dp[0][1] = 0;//第一天不持有股票时所得最多现金
        for (int i =1; i < dp.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i] - 2);
        }
        return dp[dp.length-1][1];
    }
}
```

思路：

1. 确定dp数组以及下标的含义

   `dp[i][j]`，第i天状态为j，所剩的最多现金为`dp[i][j]`。

   状态：

   - 状态一：持有股票
   - 状态二：不持有股票

2. 确定递推公式

   ```java
   		for (int i =1; i < dp.length; i++){
               dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
               dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i] - 2);
           }
   ```

   

**运行速度完全不如贪心，但是逻辑比贪心简单多了，在有动态规划dp数组框架的情况下，非常容易写出来。**













## ----子序列问题

## [0300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)（新动态规划的思路）

```java
//0300. 最长递增子序列
public class Solution26 {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j <= i - 1; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        int max = -1;
        for (int i = 0; i < dp.length; i++) {
            max = max >= dp[i] ? max : dp[i];
        }
        return max;
    }
}
```

思路：

1. dp[i]的定义

   **dp[i]表示i之前包括i的以nums[i]结尾最长上升子序列的长度**

2. 状态转移方程

   位置i的最长升序子序列等于j从0到i-1各个位置的最长升序子序列 + 1 的最大值。

   所以：if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);

   **注意这里不是要dp[i] 与 dp[j] + 1进行比较，而是我们要取dp[j] + 1的最大值**。

3. dp[i]的初始化

   每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1。



可以调试一下走一遍这个双层的遍历，初始化要把所有元素的值都设置为1，因为在还没开始遍历前，它们自己都是子序列的其中一个一个元素，所以最小值就是1。

这个双层遍历里的第二层遍历，实际上就是找出0到i-1中最大的那个子序列，然后再给它加1赋值到当前dp数组元素。



```java
// Dynamic programming + Dichotomy.
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int res = 0;
        for(int num : nums) {
            int i = 0, j = res;
            while(i < j) {
                int m = (i + j) / 2;
                if(tails[m] < num) i = m + 1;
                else j = m;
            }
            tails[i] = num;
            if(res == j) res++;
        }
        return res;
    }
}
```

**解法二：动态规划+二分查找**

思路：

- **降低复杂度切入点：** 解法一中，遍历计算 dp 列表需 O(N)，计算每个 dp[k] 需 O(N)。

  1. 动态规划中，通过线性遍历来计算 dp的复杂度无法降低
  2. 每轮计算中，需要通过线性遍历 [0,k) 区间元素来得到 dp[k] 。我们考虑：是否可以通过重新设计状态定义，使整个 dp 为一个排序列表；这样在计算每个 dp[k] 时，就可以通过二分法遍历 [0,k)区间元素，将此部分复杂度由 O(N)降至 O(logN)

- **设计思路：**

  - **新的状态定义：**
    - 我们考虑维护一个列表 tails，其中每个元素 tails[k] 的值代表 **长度为 k+1 的子序列尾部元素的值**。
    - 如 [1,4,6] 序列，长度为 1,2,3 的子序列尾部元素值分别为 tails = [1,4,6]。
  - **状态转移设计：**
    - 设常量数字 N，和随机数字 x，我们可以容易推出：当 N 越小时，N<x 的几率越大。例如： N=0 肯定比 N=1000 更可能满足 N<x。
    - 在遍历计算每个 tails[k]，不断更新长度为 [1,k] 的子序列尾部元素值，始终保持每个尾部元素值最小 （例如 [1,5,3]， 遍历到元素 5 时，长度为 2 的子序列尾部元素值为 5；当遍历到元素 3 时，尾部元素值应更新至 3，因为 3 遇到比它大的数字的几率更大)
  - tails 列表一定是严格递增的:即当尽可能使每个子序列尾部元素值最小的前提下，子序列越长，其序列尾部元素值一定更大。
    - 反证法证明： 当 k < i，若 tails[k] >= tails[i]，代表较短子序列的尾部元素的值 > 较长子序列的尾部元素的值。这是不可能的，因为从长度为 i 的子序列尾部倒序删除 i-1 个元素，剩下的为长度为 k 的子序列，设此序列尾部元素值为 v，则一定有 v<tails[i]（即长度为 k 的子序列尾部元素值一定更小）， 这和 tails[k]>=tails[i] 矛盾。
    - 既然严格递增，每轮计算 tails[k] 时就可以使用二分法查找需要更新的尾部元素值的对应索引 i。

- **算法流程：**

  - **状态定义：**
    - Tails[k] 的值代表 长度为 k+1 子序列 的尾部元素值。
  - **转移方程：**设 res 为 tails 当前长度，代表直到当前的最长上升子序列长度。设 j∈[0,res)，考虑每轮遍历 nums[k] 时，通过二分法遍历 [0,res) 列表区间，找出 nums[k] 的大小分界点，会出现两种情况：
    - 区间中存在 tails[i] > nums[k]： 将第一个满足 tails[i] > nums[k] 执行 tails[i] = nums[k]；因为更小的 nums[k] 后更可能接一个比它大的数字（前面分析过）。
    - 区间中不存在 tails[i] > nums[k]： 意味着 nums[k] 可以接在前面所有长度的子序列之后，因此肯定是接到最长的后面（长度为 res ），新子序列长度为 res + 1。
  - **初始状态：**
    - 令 tails 列表所有值 =0。
  - **返回值：**
    - 返回 res ，即最长上升子子序列长度。

  ![最长递增子序列](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/%E6%9C%80%E9%95%BF%E9%80%92%E5%A2%9E%E5%AD%90%E5%BA%8F%E5%88%97.gif)











## [0674. 最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

```java
//0674.最长连续递增序列
public class Solution27 {
    public int findLengthOfLCIS(int[] nums) {
        int[] dp =new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < dp.length; i++) {
            if (nums[i] > nums[i - 1]) {
                dp[i] = dp[i - 1] + 1;
            }
        }
        int max = 0;
        for (int i = 0; i < dp.length; i++) {
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```

1. dp[i]的定义

   **dp[i]表示i之前包括i的以nums[i]结尾最长上升子序列的长度**

2. 状态转移方程

   因为必须是连续递增序列，所以这次的递推只需要考虑前一位元素即可，不需要和0674.最长连续递增序列一样，把前面的元素都遍历一遍找最大值。

   ```java
   			if (nums[i] > nums[i - 1]) {
                   dp[i] = dp[i - 1] + 1;
               }
   ```

3. dp[i]的初始化

   每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1。









## [0718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

```java
//0718.最长重复子数组
class Solution 28{
    public int findLength(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        dp[0][0] = 0;
        int max = 0;
        for (int i = 1; i < nums1.length + 1; i++) {
            for (int j = 1; j < nums2.length + 1; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return max;
    }
}
```

题目：

动态规划思想是希望连续的，也就是说上一个状态和下一个状态(自变量)之间有关系而且连续。

公共子数组相当于子串：是**连续的**

`dp[i][j]`：表示第一个数组 A 前 i 个元素和数组 B 前 j 个元素组成的最长公共子数组(相当于子串)的长度。

我们在计算 `dp[i][j]` 的时候：

1. 若当前两个元素值相同，即 A[i] == B[j]，则说明当前元素可以构成公共子数组，所以还要加上它们的前一个元素构成的最长公共子数组的长度(在原来的基础上加 1)，此时状态转移方程：`dp[i][j] = dp[i - 1][j - 1] + 1`。

2. 若当前两个元素值不同，即 A[i] != B[j]，则说明当前元素无法构成公共子数组(就是：当前元素不能成为公共子数组里的一员)。因为公共子数组必须是连续的，而此时的元素值不同，相当于直接断开了，此时状态转移方程：`dp[i][j] = 0`。






给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

思路：

1. 确定dp数组（dp table）以及下标的含义

   `dp[i][j]` ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为`dp[i][j]`。

2. 确定递推公式

   ```java
   if (nums1[i] == nums2[j]) {
       dp[i][j] = d[i - 1][j - 1] + 1
   }
   ```

3. dp数组如何初始化

   默认初始所有元素都应该为0，因为一开始没有任何重复元素。







## [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

```java
//1143. 最长公共子序列
public class Solution29 {
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        dp[0][0] = 0;
        int max = 0;
        for (int i = 1; i < text1.length() + 1; i++) {
            for (int j = 1; j < text2.length() + 1; j++) {
                dp[i][j] = Math.max(dp[i - 1][j - 1] + (text1.charAt(i - 1) == text2.charAt(j - 1) ? 1 : 0), Math.max(dp[i][j - 1], dp[i - 1][j]));
                max = Math.max(dp[i][j], max);
            }
        }
        return max;
    }
}
```

思路：

1. 确定dp数组（dp table）以及下标的含义

   `dp[i][j]` ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为`dp[i][j]`。

2. 确定递推公式

   本题与找公共子数组不一样在，数组必须是连续，而子序列的话，可以不是连续的。

   所以从动态的角度，实际上每一次`dp[i][j]`的更新，都会影响到其同一列以及同一行的剩下元素，举个例子就是：`dp[i + n][j] = dp[i][j]`、`dp[i][j + n] = dp[i][j]`。那么代码实现可以是：

   ```java
   dp[i][j] = Math.max(dp[i - 1][j - 1] + (text1.charAt(i - 1) == text2.charAt(j - 1) ? 1 : 0), Math.max(dp[i][j - 1], dp[i - 1][j]));
   ```

   **==以下代码逻辑是不对的：==**

   ```java
   				if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                       dp[i][j] = Math.max(dp[i - 1][j], Math.max(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                   } else {
                       dp[i][j] = dp[i - 1][j - 1];
                   }
   ```

   这里实际上是有优化的：

   ```java
   if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
   	dp[i][j] = dp[i - 1][j - 1] + 1;
   }else {
   	dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
   }
   ```

3. dp数组如何初始化

   初始化都为0，因为没有元素那么公共子序列就为0。











## [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足：nums1[i] == nums2[j] 且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

```java
//1035. 不相交的线
public class Solution30 {
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        dp[0][0] = 0;
        int max = 0;
        for (int i = 1; i < nums1.length + 1; i++) {
            for (int j = 1; j < nums2.length + 1; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
                max = Math.max(dp[i][j], max);
            }
        }
        return max;
    }
}
```

思路：

因为不能有相交，所以可以理解为连线所连出的连续子序列。

> 直线不能相交，这就是说明在字符串A中 找到一个与字符串B相同的子序列，且这个子序列不能改变相对顺序，只要相对顺序不改变，链接相同数字的直线就不会相交。

那么这题就等价于1143.最长公共子序列了。



**代码优化**：

代码优化
上面代码中计算当前值的时候只会和左边，上边，左上边这3个位置的值有关，和其他的值无关，所以没必要使用二维数组，我们可以改为一维数组，怎么改我们画个图看一下

如果只是`dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j])`，我们直接把前面的一维去掉即可，其他的不需要修改，即`dp[j] = Math.max(dp[j - 1], dp[j])`;，如下图所示，

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/1621563135-woqbBB-image.png)

如果`dp[i][j] = dp[i - 1][j - 1] + 1`，上面方式就行不通了，如果改为`dp[j] = dp[j - 1] + 1`，我们发现这个`dp[j - 1]` 并不是之前的那个`dp[j - 1]`，在前一步计算的时候已经被覆盖掉了，所以我们需要一个变量在计算`dp[j - 1]`的值之前先要把dp[j - 1]的值给存起来，如下图所示

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/1621563151-WqjVnr-image.png)

```java
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int dp[] = new int[n + 1];
        for (int i = 1; i <= m; ++i) {
            int last = dp[0];
            for (int j = 1; j <= n; ++j) {
                //dp[j]计算过会被覆盖，这里先把他存储起来
                int temp = dp[j];
                //下面是递推公式
                if (nums1[i - 1] == nums2[j - 1])
                    dp[j] = last + 1;
                else
                    dp[j] = Math.max(dp[j - 1], dp[j]);
                last = temp;
            }
        }
        return dp[n];
    }
```













## [0053. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

```java
//0053. 最大子数组和
public class Solution31 {
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int max = dp[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            max = max > dp[i] ? max : dp[i];
        }
        return max;
    }
}
```











## [0115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

```java
//0115. 不同的子序列
public class Solution32 {
    public int numDistinct(String s, String t) {
        int[][][] dp = new int[t.length() + 1][s.length() + 1][2];
        for (int i = 0; i < s.length() + 1; i++) {
            dp[0][i][1] = 1;
        }
        dp[0][0][0] = 0;//当前最大子序列长度
        dp[0][0][1] = 1;//记录组合数
        for (int i = 1; i < t.length() + 1; i++) {
            for (int j = 1; j < s.length() + 1; j++) {
                if (t.charAt(i - 1) == s.charAt(j - 1)) {
                    dp[i][j][0] = dp[i - 1][j - 1][0] + 1;
                    dp[i][j][1] = dp[i - 1][j - 1][1] + dp[i][j - 1][1];
                }else {
                    dp[i][j][0] = Math.max(dp[i][j - 1][0], dp[i - 1][j][0]);
                    dp[i][j][1] = dp[i][j - 1][1];
                }
            }
        }
        if (dp[dp.length - 1][dp[0].length - 1][0] != t.length()) {
            return 0;
        }
        return dp[dp.length - 1][dp[0].length - 1][1];
    }
}
```

思路：

1. 确定dp数组（dp table）以及下标的含义

   `dp[i][j][0]`代表的是当前在i-1和j-1时的最大子序列长度

   `dp[i][j][1]`代表的是记录的排列数

2. 确定递推公式

   首先`dp[i][j][0]`是和之前一样递推公式为：

   ```java
   if (t.charAt(i - 1) == s.charAt(j - 1)) {
   	dp[i][j][0] = dp[i - 1][j - 1][0] + 1;
   }else {
   	dp[i][j][0] = Math.max(dp[i][j - 1][0], dp[i - 1][j][0]);
   }
   ```

   而`dp[i][j][1]`是由`dp[i - 1][j - 1][1]`和`dp[i][j - 1][1]`一起确定的：

   ```
   if (t.charAt(i - 1) == s.charAt(j - 1)) {
   	dp[i][j][1] = dp[i - 1][j - 1][1] + dp[i][j - 1][1];
   }else {
   	dp[i][j][1] = dp[i][j - 1][1];
   }
   ```

   <img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/IMG_0751.PNG" alt="IMG_0751" style="zoom:50%;" />

3. 初始化

   应该将i=0时的所有元素即`dp[0][j][1]`初始化为1，为后续递推做准备。

4. 遍历顺序

   第一层遍历数组s，第二层再遍历数组t（t为s的子序列），看上图可以得知为什么要这样的遍历顺序。











## [0583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

```java
//0583. 两个字符串的删除操作
public class Solution33 {
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 1; i < word1.length() + 1; i++) {
            for (int j = 1; j < word2.length() + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        int max = dp[dp.length - 1][dp[0].length - 1];
        return word1.length() + word2.length() - 2 * max;
    }
}
```

需要得到本题答案，那么就首先变相去计算最长公共子序列，因为最终答案就是（word1的长度减去最大公共子序列的长度）加上（word2的长度减去最大公共子序列的长度）。

参考1143.最长公共子序列即可。不做详细解释了。









## [0072. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)(坚持刷题的结果就是hard题都是小case^_^)

```java
//0072. 编辑距离
public class Solution34 {
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word2.length() + 1][word1.length() + 1];
        for (int i = 0; i < dp.length; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < dp[0].length; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (word2.charAt(i - 1) == word1.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + 1, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
}
```

思路：

1. 确定dp数组

   定义`dp[i][j]`为遍历到word2的i - 1位置以及word1的j - 1位置时的最少操作次数（操作解释）：

   - 插入一个字符
   - 删除一个字符
   - 替换一个字符

2. 确定递推公式

   ```java
   				if (word2.charAt(i - 1) == word1.charAt(j - 1)) {
                       dp[i][j] = dp[i - 1][j - 1];
                   }
   ```

   如果`word2.charAt(i - 1) == word1.charAt(j - 1)`，那么在当下两个子数组中，两个字符不仅位置都处于末尾（不需要执行额外的删除或者插入操作），并且字符一样（不需要执行替换操作）所以不需要做出任何操作，只需要进行`dp[i - 1][j - 1]`时相应的操作即可。

   ```java
   else {
   	dp[i][j] = Math.min(dp[i - 1][j - 1] + 1, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
   }
   ```

   如果`word2.charAt(i - 1) != word1.charAt(j - 1)`，那么这时候有三条路可以走到`dp[i][j]`：

   - `dp[i - 1][j - 1] + 1`
   - `dp[i - 1][j] + 1`
   - `dp[i][j - 1] + 1`

   取这三个值得最小值，就是`dp[i][j]`

3. 初始化

   ```java
   		for (int i = 0; i < dp.length; i++) {
               dp[i][0] = i;
           }
           for (int j = 0; j < dp[0].length; j++) {
               dp[0][j] = j;
           }
   ```

   考虑到递推公式，所以要先把i=0这一行，以及j=0这一列初始化，实际上可以理解为，当字符串s1从0慢慢变大到word，这个过程中对比的字符串始终是一个空的字符串，那么s1要变成空字符串要做的操作次数就是其自身长度，要删除自身长度个字符。











## [0647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```java
//0647. 回文子串
public class Solution35 {
    public int countSubstrings(String s) {
        Boolean[][] dp = new Boolean[s.length()][s.length()];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                dp[i][j] = false;
            }
        }
        for (int i = dp.length - 1; i >= 0; i--) {
            for(int j = i; j < dp[0].length; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                    }else if (dp[i + 1][j - 1]) {
                        dp[i][j] = true;
                    }
                }
            }
        }
        int count = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (dp[i][j]) {
                    count++;
                }
            }
        }
        return count;
    }
}
```

思路：

1. 确定dp数组（dp table）以及下标的含义

   布尔类型的dp[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp[i][j]为true，否则为false。

2. 确定递推公式

   在确定递推公式时，就要分析如下几种情况。

   整体上是两种，就是s[i]与s[j]相等，s[i]与s[j]不相等这两种。

   当s[i]与s[j]不相等，那没啥好说的了，dp[i][j]一定是false。

   当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况

   - 情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
   - 情况二：下标i 与 j相差为1，例如aa，也是文子串
   - 情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，那么aba的区间就是 i+1 与 j-1区间，这个区间是不是回文就看`dp[i + 1][j - 1]`是否为true。

3. 确定遍历顺序

   如果这矩阵是从上到下，从左到右遍历，那么会用到没有计算过的`dp[i + 1][j - 1]`，也就是根据不确定是不是回文的区间[i+1,j-1]，来判断了[i,j]是不是回文，那结果一定是不对的。

   **所以一定要从下到上，从左到右遍历，这样保证`dp[i + 1`][j - 1]都是经过计算的**。

   有的代码实现是优先遍历列，然后遍历行，其实也是一个道理，都是为了保证`dp[i + 1][j - 1]`都是经过计算的。



思路2：

中心扩散法，利用双指针。

```java
	public int countSubstrings(String s) {
        int len, ans = 0;
        if (s == null || (len = s.length()) < 1) return 0;
        //总共有2 * len - 1个中心点
        for (int i = 0; i < 2 * len - 1; i++) {
            //通过遍历每个回文中心，向两边扩散，并判断是否回文字串
            //有两种情况，left == right，right = left + 1，这两种回文中心是不一样的
            int left = i / 2, right = left + i % 2;
            while (left >= 0 && right < len && s.charAt(left) == s.charAt(right)) {
                //如果当前是一个回文串，则记录数量
                ans++;
                left--;
                right++;
            }
        }
        return ans;
    }
```











## [0516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

```java
//0516. 最长回文子序列
public class Solution36 {
    public int longestPalindromeSubseq(String s) {
        int[][] dp = new int[s.length()][s.length()];
        int max = 0;
        for (int i = dp.length - 1; i >= 0; i--) {
            for(int j = i; j < dp[0].length; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j == i) {
                        dp[i][j] = 1;
                    }else if(j - i == 1) {
                        dp[i][j] = 2;
                    }else {
                        dp[i][j] = dp[i + 1][j - 1] + 2;
                    }
                }else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return max;
    }
}
```

1. dp数组定义

   `dp[i][j]`指的是子字符串[i, j]中所包含的最长回文子序列。

2. 确定地推公式

   当`s.charAt(i) == s.charAt(j)`时，有三种情况：

   - i==j时，说明此时就由一个字母组成，则`dp[i][j]=1`
   - j - i == 1时，说明此时由两个字母组成，且相同，则`dp[i][j]=2`
   - 差值大于等于2时`dp[i][j] = dp[i + 1][j - 1] + 2;`

   当`s.charAt(i) != s.charAt(j)`时：

   则`dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1])`

3. 遍历顺序，因为要用到`dp[i+1][j-1]`所以遍历i应该从后往前遍历。



本题还有优化，即在`s.charAt(i) == s.charAt(j)`的条件里，把`j==i`和`j-i==1`两种判断条件在初始化dp数组时完成。

```java
for (int i = 0; i < dp.length; i++) {
    dp[i][i] = 1;
}
```





## ----补充问题

## [0221. 最大正方形](https://leetcode.cn/problems/maximal-square/)

```java
	public int maximalSquare(char[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int max = 0;
        for (int j = 0; j < dp[0].length; j++) {
            if (matrix[0][j] == '0') {
                dp[0][j] = 0;
            }else {
                dp[0][j] = 1;
                max = 1;
            }
        }
        for (int i = 0; i < dp.length; i++) {
            if (matrix[i][0] == '0') {
                dp[i][0] = 0;
            }else {
                dp[i][0] = 1;
                max = 1;
            }
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (matrix[i][j] == '0') {
                    dp[i][j] = 0;
                }else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                    max = Math.max(max, dp[i][j] * dp[i][j]);
                }
            }
        }
        return max;
    }
```

动态规划：

可以使用动态规划降低时间复杂度。我们用dp(i,j)表示以 (i, j) 为右下角，且只包含 1 的正方形的边长最大值。如果我们能计算出所有dp(i,j)的值，**那么其中的最大值即为矩阵中只包含 1 的正方形的边长最大值**，==其平方即为最大正方形的面积==。

那么如何计算dp中的每个元素值呢？对于每个位置 (i, j)，检查在矩阵中该位置的值：

1. 如果该位置的值是 0，则 dp(i, j) = 0，因为当前位置不可能在由 1 组成的正方形中；

2. 如果该位置的值是 1，则 dp(i,j) 的值由其上方、左方和左上方的三个相邻位置的dp 值决定。具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1，状态转移方程如下：

   `dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])`







## [1277. 统计全为 1 的正方形子矩阵](https://leetcode.cn/problems/count-square-submatrices-with-all-ones/)

```java
	public int countSquares(int[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int sum = 0;
        //先把第一列和第一行进行预处理，但是不要重复扫描(0,0)
        for (int j = 0; j < dp[0].length; j++) {
            if (matrix[0][j] == 0) {
                dp[0][j] = 0;
            } else {
                dp[0][j] = 1;
                sum += 1;
            }
        }
        //避免重复扫描(0,0)
        for (int i = 1; i < dp.length; i++) {
            if (matrix[i][0] == 0) {
                dp[i][0] = 0;
            } else {
                dp[i][0] = 1;
                sum += 1;
            }
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                    sum += dp[i][j];
                }
            }
        }
        return sum;
    }
```

动态规划：

和0221.最大正方形的做法一样，先去构造dp数组，用dp(i,j)表示以 (i, j) 为右下角，且只包含 1 的正方形的边长最大值

那么如何计算dp中的每个元素值呢？对于每个位置 (i, j)，检查在矩阵中该位置的值：

1. 如果该位置的值是 0，则 dp(i, j) = 0，因为当前位置不可能在由 1 组成的正方形中；

2. 如果该位置的值是 1，则 dp(i,j) 的值由其上方、左方和左上方的三个相邻位置的dp 值决定。具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1，状态转移方程如下：

   `dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])`

dp(i,j)如果为2，说明从(i,j)开始往左上圈，可以圈出一个边长为1和边长为2的正方形，同理，若dp(i,j)=n，则以(i,j)开始往左上圈正方形可以圈出n个正方形，则最后遍历dp数组，相加所有dp(i,j)的value，就是答案。







## [0032. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

```java
	public int longestValidParentheses(String s) {
        int maxans = 0;
        int[] dp = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }
```

我们定义dp[i] 表示以下标 i字符结尾的最长有效括号的长度。我们将dp 数组全部初始化为 0 。显然有效的子串一定以 ‘)’ 结尾，因此我们可以知道以 ‘(’ 结尾的子串对应的 dp 值必定为 0，我们只需要求解 ‘)’ 在 dp 数组中对应位置的值。

1. `s[i] = ')'且s[i - 1] = '('`，当字符串型如".....()"，我们可以推出：

   ​													`dp[i] = dp[i - 2] + 2`

   我们可以进行这样的转移，是因为结束部分的 "()" 是一个有效子字符串，并且将之前有效子字符串的长度增加了 2.

2. `s[i] = ')'且s[i - 1] = ')'`，也就是字符串如"......))"，我们可以推出：

   如果s[i - dp[i - 1] - 1] = '('，那么

   ​													`dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 1] + 2`



我们考虑如果倒数第二个 ‘)’ 是一个有效子字符串的一部分（记作subs），对于最后一个‘)’ ，如果它是一个更长子字符串的一部分，那么它一定有一个对应的 ‘(’ ，且它的位置在倒数第二个 ‘)’ 所在的有效子字符串的前面（也就是subs的前面）。因此，如果子字符串subs的前面恰好是 ‘(’ ，那么我们就用2加上subs的长度(dp[i−1])去更新dp[i]。同时，我们也会把有效子串 “(subs)” 之前的有效子串的长度也加上，也就是再加上 dp[i−dp[i−1]−2]。

最后的答案即为dp 数组中的最大值。





## [0064. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

```java
public int minPathSum(int[][] grid) {
    int[][] dp = new int[grid.length][grid[0].length];
    dp[0][0] = grid[0][0];
    for (int j = 1; j < dp[0].length; j++) {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    for (int i = 1; i < dp.length; i++) {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for (int i = 1; i < dp.length; i++) {
        for (int j = 1; j < dp[0].length; j++) {
            dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[dp.length - 1][dp[0].length - 1];
}
```

在一个`m*n`网格，找出一个从左上角到右下角的路径，每次只能向下或者向右移动一步，因此这个动态存储的过程就非常清晰：

1. 从上往下走`dp[i][j] = dp[i - 1][j] + grid[i][j]`
2. 从左往右走`dp[i][j] = dp[i][j - 1] + grid[i][j]`

因此`dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1])`









## [0152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

```java
public int maxProduct(int[] nums) {
    int maxF = nums[0], minF = nums[0], ans = nums[0];
    int length = nums.length;
    for (int i = 1; i < length; ++i) {
        int mx = maxF, mn = minF;
        maxF = Math.max(mx * nums[i], Math.max(nums[i], mn * nums[i]));
        minF = Math.min(mn * nums[i], Math.min(nums[i], mx * nums[i]));
        ans = Math.max(maxF, ans);
    }
    return ans;
}
```

动态规划：

这题是求数组中子区间的最大乘积，对于乘法，我们需要注意，负数乘以负数，会变成正数，所以解这题的时候我们需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。

我们的动态方程可能这样：

`maxDP[i + 1] = max(maxDP[i] * A[i + 1], A[i + 1],minDP[i] * A[i + 1])`
`minDP[i + 1] = min(minDP[i] * A[i + 1], A[i + 1],maxDP[i] * A[i + 1])`
`dp[i + 1] = max(dp[i], maxDP[i + 1])`









## [0312. 戳气球](https://leetcode.cn/problems/burst-balloons/)

有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。

解法一：递归（但是超时）

```java
int max = 0;
int[] input;
public int maxCoins(int[] nums) {
    if (nums.length == 1) {
        return nums[0];
    }
    input = nums;
    findMax(-1, nums.length);
    return max;
}

private int findMax(int left, int right) {
    if (left + 2 == right) {
        return (left == -1 ? 1 : input[left]) * input[left + 1] * (right == input.length ? 1 : input[right]);
    }
    int curMax = 0;
    for (int i = left + 1; i < right; i++) {
        int total = findMax(left, i) + findMax(i, right) + (left == -1 ? 1 : input[left]) * input[i] * (right == input.length ? 1 : input[right]);
        curMax = Math.max(curMax, total);
    }
    max = Math.max(max, curMax);
    return curMax;
}
```

解法二：动态规划

```java
public int maxCoins3(int[] nums) {
    int[][] dp = new int[nums.length + 2][nums.length + 2];//数组长度为nums.length + 2是因为左边界和右边界在本题也被处理
    int[] val = new int[nums.length + 2];
    val[0] = val[nums.length + 1] = 1;
    for (int i = 1; i < val.length - 1; i++) {
        val[i] = nums[i - 1];
    }
    for (int i = dp.length - 3; i >= 0; i--) {
        for (int j = i + 2; j < dp.length; j++) {
            for (int k = i + 1; k < j; k++) {
                int sum = dp[i][k] + dp[k][j] + val[k - 1] * val[k] * val[k + 1];
                dp[i][j] = Math.max(dp[i][j], sum);
            }
        }
    }
    return dp[0][dp.length - 1];
}
```

不管是递归也好还是动态规划，其推进公式都是：

![image-20220901115815457](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220901115815457.png)

`dp[i][j]`记录的是在开区间[i,j]能得到的最多硬币数











## [1478. Allocate Mailboxes](https://leetcode.cn/problems/allocate-mailboxes/)

```
median = i + (houses.length - i) / 2;
X  X  X  X  0  X  X  X
   X  X  X  0  X  X  X
      X  X  X  0  X  X
      	 X  X  0  X  X
```

```
dp[i][k]: minimum total distance between each house and its	nearest mailbox for houses[0:i] coverd by k mailboxes
[XXXXXXXXX] [XXXX i]
     k-1 ^   ^
         j  j+1
dp[i][k] = min{dp[j][k-1] + range[j+1][i]} for j = ...
```

```java
//1478. Allocate Mailboxes
public class Solution43 {
    int[][] range;

    public int minDistance(int[] houses, int k) {
        int[][] dp = new int[houses.length][k + 1];
        Arrays.sort(houses);
        range = new int[houses.length][houses.length];
        for (int i = 0; i < houses.length; i++) {
            for (int j = i; j < houses.length; j++) {
                range[i][j] = 0;
                for (int kk = i; kk <= j; kk++) {
                    range[i][j] += Math.abs(houses[kk] - houses[(i + j) / 2]);
                }
            }
        }
        for (int i = 0; i < houses.length; i++) {
            dp[i][1] = range[0][i];
        }
        for (int i = 0; i < houses.length; i++) {
            for (int kk = 2; kk < dp[0].length; kk++) {
                //这里不能用MAX_VALUE，如果是j >= i时，则dp[i][kk]会直接变成MAX，此时再做加法就会变为负数，会影响到下面取最小值的结果
                dp[i][kk] = 10001;
                for (int j = 0; j < i; j++) {
                    dp[i][kk] = Math.min(dp[i][kk], dp[j][kk - 1] + range[j + 1][i]);
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
}
```













# Monotonic Stack

**通常是一维数组，要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置，此时我们就要想到可以用单调栈了**。

那么单调栈的原理是什么呢？为什么时间复杂度是O(n)就可以找到每一个元素的右边第一个比它大的元素位置呢？

单调栈的本质是空间换时间，因为在遍历的过程中需要用一个栈来记录右边第一个比当前元素的元素，优点是只需要遍历一次。

在使用单调栈的时候首先要明确如下几点：

1. 单调栈里存放的元素是什么？

   单调栈里只需要存放元素的下标i就可以了，如果需要使用对应的元素，直接T[i]就可以获取。

2. 单调栈里元素是递增呢？ 还是递减呢？

   **注意一下顺序为 从栈头到栈底的顺序**

使用单调栈主要有三个判断条件。

- 当前遍历的元素T[i]小于栈顶元素T[st.top()]的情况
- 当前遍历的元素T[i]等于栈顶元素T[st.top()]的情况
- 当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况

## [0739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

```java
//0739. 每日温度
public class Solution1 {
    public int[] dailyTemperatures(int[] temperatures) {
        int[] answer = new int[temperatures.length];
        Deque<Integer> deque = new LinkedList<>();
        deque.addLast(0);
        for (int i = 1; i < temperatures.length; i++) {
            if (temperatures[i] > temperatures[deque.peekLast()]) {
                while (!deque.isEmpty()) {
                    if (temperatures[i] > temperatures[deque.peekLast()]) {
                        int index = deque.pollLast();
                        answer[index] = i - index;
                    }else {
                        break;
                    }
                }
            }
            deque.addLast(i);
        }
        return answer;
    }
}
```

本题实际上只要考虑当前遍历元素T[i]大于栈顶元素T[st.top()]的情况：

```java
			if (temperatures[i] > temperatures[deque.peekLast()]) {
                while (!deque.isEmpty()) {
                    if (temperatures[i] > temperatures[deque.peekLast()]) {
                        int index = deque.pollLast();
                        answer[index] = i - index;
                    }else {
                        break;
                    }
                }
            }
```

当发现当前遍历元素T[i]大于栈顶元素T[st.top()]时，进入第二层遍历，从栈顶到栈尾的遍历顺序，直到碰到遍历元素小于栈顶元素时break。

因为栈里存放的是元素下标，所以也可以由此求出两个元素的中间间隔即`i - index`。











## [0496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)

```java
//0496. 下一个更大元素 I
public class Solution2 {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] answer = new int[nums1.length];
        //因为本题不会出现重复元素，所以用hashmap的key来记录nums1中存在的元素，value就是其下标index
        Map<Integer, Integer> hashmap = new HashMap<>();
        //用deque来当nums2的单调栈，存放元素下标
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < nums1.length; i++) {
            if (!hashmap.containsKey(nums1[i])) {
                hashmap.put(nums1[i], i);
                answer[i] = -1;
            }
        }
        deque.addLast(0);
        for (int i = 1; i < nums2.length; i++) {
            while (!deque.isEmpty() && nums2[i] > nums2[deque.peekLast()]) {
                //deque中存放的是nums2的元素下标，所以其值是nums2[deque.peekLast()]
                int index = deque.pollLast();
                if (hashmap.containsKey(nums2[index])) {
                    //本题是要获得第一个比x大的元素值（非距离）
                    answer[hashmap.get(nums2[index])] = nums2[i];
                }
            }
            deque.addLast(i);
        }
        return answer;
    }
}
```

与0739.每日温度相比，本题多了一些条件，而核心还是找右侧第一个比x大的元素（不是元素下标），看上面代码标注部分就是要修改的一些地方。











## [0503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

```java
//0503. 下一个更大元素 II
public class Solution3 {
    public int[] nextGreaterElements(int[] nums) {
        int[] answer = new int[nums.length];
        for (int i = 0; i < answer.length; i++) {
            answer[i] = -1;
        }
        Deque<Integer> deque = new LinkedList<>();
        deque.addLast(0);
        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[i] > nums[deque.peekLast()]) {
                int index = deque.pollLast();
                answer[index] = nums[i];
            }
            deque.addLast(i);
        }
        int last = deque.peekLast();
        for (int i = 0; i < last; i++) {
            while (!deque.isEmpty() && nums[i] > nums[deque.peekLast()]) {
                int index = deque.pollLast();
                answer[index] = nums[i];
            }
        }
        return answer;
    }
}
```

题目：

给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。

数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。



本题是一个循环数组，也就是说遍历到nums.length-1时并不是结束，还需要继续循环遍历。

本人是采用两次循环，来让遍历走完一个完整的流程。

1. 先正常遍历到nums.length - 1，这个时候deque中一定还存在未找到更大元素的元素在栈内，而deque中存放的又是元素的index下标，所以这时候deque.peekLast()这个值就是我们第二轮循环要遍历到的位置（其实就是nums.length()-1，所以直接再遍历一次nums即可，因为最后一个元素一定会留着栈里）。
2. 第二次遍历我们就不需要再往deque中放元素了。











## [042. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

```java
//0042. 接雨水
public class Solution4 {
    //单调栈
    public int trap(int[] height) {
        Deque<Integer> deque = new LinkedList<>();
        deque.addLast(0);
        int count = 0;
        for (int i = 1; i < height.length; i++) {
            while (!deque.isEmpty() && height[i] >= height[deque.peekLast()]) {
                int mid = deque.pollLast();
                //这一步很关键，相当于先把middle，也就是凹槽底部这个元素出栈并记录，然后再去找它左边的元素得到左边高度
                if (!deque.isEmpty()) {
                    int index = deque.peekLast();
                    int wide = i - index - 1;
                    int h = Math.min(height[index], height[i]) - height[mid];
                    count += h * wide;
                }
            }
            deque.addLast(i);
        }
        return count;
    }
    //双指针法
    public int trap1(int[] height) {
        int count = 0;
        for (int i = 0; i < height.length; i++) {
            if (i == 0 || i == height.length - 1) {
                continue;
            }
            int maxLeft = height[i];
            int maxRight = height[i];
            for (int j = i - 1; j >= 0; j--) {
                maxLeft = Math.max(maxLeft, height[j]);
            }
            for (int j = i + 1; j < height.length; j++) {
                maxRight = Math.max(maxRight, height[j]);
            }
            count += (maxLeft < maxRight ? maxLeft : maxRight) - height[i];
        }
        return count;
    }
    //动态规划
    public int trap2(int[] height) {
        int[] maxLeftDp = new int[height.length];
        int[] maxRightDp = new int[height.length];
        maxLeftDp[0] = height[0];
        maxRightDp[height.length - 1] = height[height.length - 1];
        //思路其实就是双指针的基础上变为动态，这样就可以省去一个for循环的嵌套
        for (int i = 1; i < height.length; i++) {
            maxLeftDp[i] = Math.max(height[i], maxLeftDp[i - 1]);
        }
        for (int i = height.length - 2; i >= 0; i--) {
            maxRightDp[i] = Math.max(height[i], maxRightDp[i + 1]);
        }
        int count = 0;
        for (int i = 1; i < height.length - 1; i++) {
            count += Math.min(maxLeftDp[i], maxRightDp[i]) - height[i];
        }
        return count;
    }
}
```

本题三种做法：

1. 双指针
2. 动态规划-基于双指针
3. 单调栈



双指针和动态规划实际上思路很简单，但是单调栈确实是比较巧妙。

我一开始用单调栈一直没想出来，因为考虑到比如[7, 1, 3, 4]这种情况，如何在遍历完3到遍历4时，动态变化1作为凹槽的积水高度（因为此时1已经从栈中出栈，那么就没法再得到这个元素的下标了）是一个问题。后来发现，其实压根不需要再管已经出栈的元素，只需要把所有积水都按照行来考虑，不需要考虑列的动态更新。

![image-20220404104656990](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220404104656990.png)

```java
//所以就有了如下代码，按照行来计算，那么我只要关心它的宽和高即可，最后积水就是h * w
			while (!deque.isEmpty() && height[i] >= height[deque.peekLast()]) {
                int mid = deque.pollLast();
                //这一步很关键，相当于先把middle，也就是凹槽底部这个元素出栈并记录，然后再去找它左边的元素得到左边高度
                if (!deque.isEmpty()) {
                    int index = deque.peekLast();
                    int wide = i - index - 1;
                    int h = Math.min(height[index], height[i]) - height[mid];
                    count += h * wide;
                }
            }
```















## [0084. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

```java
//0084. 柱状图中最大的矩形
public class Solution5 {
    //动态规划-超出内存限制
    public int largestRectangleArea(int[] heights) {
        int[][][] dp = new int[heights.length + 1][heights.length + 1][2];
        int max = 0;
        for (int i = 1; i < heights.length + 1; i++) {
            dp[i][i][0] = heights[i - 1];//用于记录到i为止最大面积
            dp[i][i][1] = heights[i - 1];//用于记录到i为止最小高度
            max = Math.max(max, dp[i][i][0]);
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = i + 1; j < dp[0].length; j++) {
                dp[i][j][1] = Math.min(dp[i][j - 1][1], heights[j - 1]);
                dp[i][j][0] = Math.max(dp[i][j - 1][0], dp[i][j][1] * (j + 1 - i));
                max = Math.max(max, dp[i][j][0]);
            }
        }
        return max;
    }
    //动态规划改一维数组-超出时间限制
    public int largestRectangleArea2(int[] heights) {
        int[][] dp = new int[heights.length + 1][2];
        int max = 0;
        for (int j = 1; j < dp.length; j++) {
            dp[j][1] = heights[j - 1];
            dp[j][0] = heights[j - 1];
            max = Math.max(max, dp[j][0]);
            for (int i = j + 1; i < dp.length; i++) {
                dp[i][1] = Math.min(dp[i - 1][1], heights[i - 1]);
                dp[i][0] = Math.max(dp[i - 1][0], dp[i][1] * (i + 1 - j));
                max = Math.max(max, dp[i][0]);
            }
        }
        return max;
    }
    //单调栈
    public int largestRectangleArea3(int[] heights) {
        Deque<Integer> deque = new LinkedList<>();
        deque.addLast(0);
        int max = 0;
        for (int i = 1; i < heights.length; i++) {
            while (!deque.isEmpty() && heights[i] < heights[deque.peekLast()]) {
                int mid = deque.pollLast();
                int w;
                if (deque.isEmpty()) {
                    w = i;
                }else {
                    w = i - deque.peekLast() - 1;
                }
                max = Math.max(max, w * heights[mid]);
            }
            deque.addLast(i);
        }
        while (!deque.isEmpty()) {
            int mid = deque.pollLast();
            int w;
            if (deque.isEmpty()) {
                w = heights.length;
            }else {
                w = heights.length - deque.peekLast() - 1;
            }
            max = Math.max(max, w * heights[mid]);
        }
        return max;
    }
    //动态规划-AC!!!
    public int largestRectangleArea4(int[] heights) {
        int[] minLeftDp = new int[heights.length];
        int[] minRightDp = new int[heights.length];
        minLeftDp[0] = -1;
        for (int i = 1; i < heights.length; i++) {
            int t = i - 1;
            while (t >= 0 && heights[t] >= heights[i]) {
                t = minLeftDp[t];
            }
            minLeftDp[i] = t;
        }
        minRightDp[heights.length - 1] = heights.length;
        for (int i = heights.length - 2; i >= 0; i--) {
            int t = i + 1;
            while (t < heights.length && heights[t] >= heights[i]) {
                t = minRightDp[t];
            }
            minRightDp[i] = t;
        }
        int result = 0;
        for (int i = 0; i < heights.length; i++) {
            int sum = heights[i] * (minRightDp[i] - minLeftDp[i] - 1);
            result = Math.max(sum, result);
        }
        return result;
    }
    //双指针-超出时间限制
    public int largestRectangleArea5(int[] heights) {
        int max = 0;
        for (int i = 0; i < heights.length; i++) {
            int left = i - 1;
            int right = i + 1;
            while (left >= 0) {
                if (heights[left] < heights[i]) {
                    break;
                }
                left--;
            }
            while (right <= heights.length - 1) {
                if (heights[right] < heights[i]) {
                    break;
                }
                right++;
            }
            max = Math.max((right - left - 1) * heights[i], max);
        }
        return max;
    }
}
```

首先先讲双指针：

找出当前节点元素左右第一个小于当前元素节点值的节点，这两个节点之间的间隔就是本题答案构成之一的宽，其高就是当前heights[i]，然后找最大的面积即可。但是运行结果是超时的。



动态规划：

我写了两种动态规划，思路是没有任何问题的，但是同样是超时，因为套了两个循环。

**==可以看题解4，这种动态规划是能成功ac的，实际思路就是动态版本的双指针也是比较好实现的，只不过这里面的dp数组存放的是当前节点左右第一个小于其本身值的元素下标，并非元素值。==**



单调栈：

本题单调栈和之前的做法都相反，单调栈里元素的排列顺序应该是从小到大（从栈尾到栈头），所以出栈的条件是：

```java
while (!deque.isEmpty() && heights[i] < heights[deque.peekLast()])
```

~~目前代码是还是有错误的~~（目前已经修改好了，在不添加首尾元素的前提下），因为本题还需要做一个非常关键的操作，就是给heights数组添加一个头元素0以及尾元素0进去，也就是说这个栈是永远不会空的（元素最小值为0，并且在数组首位，一定不会被出栈），这样就可以避免例如：[2, 1, 2]这种情况，在添加了一个头元素0之后该数组变为[0, 2, 1, 2]这样，这样可以让1这个元素的最大面积得到正确计算。在[2, 1, 2]时2先出栈后，就没有了能够记录1这个元素左边最小元素的位置了，而加入了首位0元素之后，就可以一直记录1的最大面积的左右两个下标值。

在修改代码后，不添加首尾元素也可以顺利实现单调栈了。首先先说添加尾元素0到heights的替代做法是在循环完之后，如果此时deque不为空，那么就是说这其中有一段序列一直在递增，所以我们要手动给他们出栈：

```java
		while (!deque.isEmpty()) {
            int mid = deque.pollLast();
            int w;
            if (deque.isEmpty()) {
                w = heights.length;
            }else {
                w = heights.length - deque.peekLast() - 1;
            }
            max = Math.max(max, w * heights[mid]);
        }
```

再说头元素0的替代方法：

```java
		for (int i = 1; i < heights.length; i++) {
            while (!deque.isEmpty() && heights[i] < heights[deque.peekLast()]) {
                int mid = deque.pollLast();
                int w;
                if (deque.isEmpty()) {
                    w = i;
                }else {
                    w = i - deque.peekLast() - 1;
                }
                max = Math.max(max, w * heights[mid]);
            }
            deque.addLast(i);
        }
```

我们对当前元素左边的元素进行判定，如果栈空了，那么就说明该元素是目前最小的元素或者说它就是头元素，但是都无所谓，我们此时的宽w就应该是i：

1. 假如它是头元素，那么此时i就是1那么宽也就是1，满足条件
2. 假如它不是头元素，那么它一定是此时最小元素，所以它的宽可以从头取到i，所以就是i

同理在处理尾部元素时也是一样：

```java
			if (deque.isEmpty()) {
                w = heights.length;
            }else {
                w = heights.length - deque.peekLast() - 1;
            }
```





建议本题用动态规划做，思路又简单同时运行效率还快。







## [0085. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/)

```java
public int maximalRectangle(char[][] matrix) {
    int max = 0;
    for (int i = 0; i < matrix.length; i++) {
        int[] heights = new int[matrix[0].length];
        for (int j = 0; j < matrix[0].length; j++) {
            int height = 0;
            for (int k = i; k >= 0; k--) {
                if (matrix[k][j] == '1') {
                    height++;
                } else {
                    break;
                }
            }
            heights[j] = height;
        }
        int[] minLeftDp = new int[heights.length];
        int[] minRightDp = new int[heights.length];
        minLeftDp[0] = -1;
        for (int j = 1; j < heights.length; j++) {
            int t = j - 1;
            while (t >= 0 && heights[t] >= heights[j]) {
                t = minLeftDp[t];
            }
            minLeftDp[j] = t;
        }
        minRightDp[heights.length - 1] = heights.length;
        for (int j = heights.length - 2; j >= 0; j--) {
            int t = j + 1;
            while (t < heights.length && heights[t] >= heights[j]) {
                t = minRightDp[t];
            }
            minRightDp[j] = t;
        }
        for (int j = 0; j < heights.length; j++) {
            int sum = heights[j] * (minRightDp[j] - minLeftDp[j] - 1);
            max = Math.max(sum, max);
        }
    }
    return max;
}
```

本题的主要计算思路需要参考[0084. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)。需要利用其中计算最大矩阵面积的方法。

再然后我们来看本题，实际上就是如下图所示

![image-20220809143538741](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220809143538741.png)

我们只需要一层层去创建对应的heights[]数组然后再用之前计算最大矩阵面积的方法去计算出每一层柱状图的最大矩阵面积最后进行比较即可。





# String & Array

## [0003. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```java
//0003. 无重复字符的最长子串
public class Solution1 {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0;
        int right = 0;
        int max = 0;
        while (right < s.length()) {
            if (!set.contains(s.charAt(right))) {
                set.add(s.charAt(right));
                max = Math.max(max, right - left + 1);
                right++;
            }else {
                max = Math.max(max, right - left);
                set.remove(s.charAt(left));
                left++;
            }
        }
        return max;
    }
}
```

思路：

滑动窗口

首先本题可以用set来判断是否当前滑动窗口中是否已经含有了相同的元素。初始left和right两个左右指针当作滑动窗口，滑动的条件如下：

1. right右指针作为滑动窗口的主动指针往右遍历，只要set中不存在新遍历到的元素，那么就加入set中，然后继续遍历。
2. left左指针作为滑动窗口的被动指针，当right遍历到一个已经存在于set的元素，那么此时left左指针开始往右遍历，删除set中遍历到的元素，一直删除到右指针此时遍历到的元素位置为止。

```java
            if (!set.contains(s.charAt(right))) {
                set.add(s.charAt(right));
                max = Math.max(max, right - left + 1);
                right++;
            }else {
                max = Math.max(max, right - left);
                set.remove(s.charAt(left));
                left++;
            }
```

==难度评级：比较简单==





## [0215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

```java
//0215. 数组中的第K个最大元素
public class Solution2 {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - 1 - k + 1];
    }
}
```

比较好奇这题为什么是个中等难度...









## [0033. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```java
//0033. 搜索旋转排序数组
public class Solution3 {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        //这里要注意一定是left<=right，可以是等于的，否则将少判断很多几个条件
        while (left <= right) {
            if (target > nums[left]) {
                left++;
                continue;
            }
            if (target < nums[right]) {
                right--;
                continue;
            }
            if (target == nums[left] || target == nums[right]) {
                return target == nums[left] ? left : right;
            }
            if (target < nums[left] && target > nums[right]) {
                return -1;
            }
        }
        return -1;
    }
}
```

思路：

利用双指针，由于数组是有序排序后进行了旋转，那么其实就是变成了两个有序数组，那么左指针就从小到大遍历第一个数组，右指针就从大到小遍历第二个数组。

首先明白一个道理，就是左数组的最小值是大于右数组的最大值的，所以此时：

1. `target > nums[left]`：左指针往右移，这个时候其实已经确定了target是在左数组上面了。
2. `target < nums[right]`：右指针往左移，此时确定了target在右数组上。
3. `target == nums[left] || target == nums[right]`：如果相等，那么就是找到了目标值。
4. `target < nums[left] && target > nums[right]`：由于左数组的最小值是大于右数组的最大值，那么有这个判断条件成立的情况就是说明数组中间有断层，不包含这个target，所以返回-1。











## [0054. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

```java
//0054. 螺旋矩阵
public class spiralOrder {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int row = matrix.length - 1;
        int col = matrix[0].length - 1;
        int tempRow = 0;
        int tempCol = 0;
        while (row >= tempRow && col >=tempCol) {
            if (row == tempRow) {
                for (int i = tempCol; i <= col - 1; i++) {
                    res.add(matrix[tempRow][i]);
                }
                break;
            }
            if (col == tempCol) {
                for (int i = tempRow; i <= row - 1; i++) {
                    res.add(matrix[i][col]);
                }
                break;
            }
            for (int i = tempCol; i <= col - 1; i++) {
                res.add(matrix[tempRow][i]);
            }
            for (int i = tempRow; i <= row - 1; i++) {
                res.add(matrix[i][col]);
            }
            for (int i = col; i >= tempCol + 1; i--) {
                res.add(matrix[row][i]);
            }
            for (int i = row; i >= tempRow + 1; i--) {
                res.add(matrix[i][tempCol]);
            }
            tempRow++;
            tempCol++;
            row--;
            col--;
        }
        return res;
    }
}
```

思路：

外圈思路很简单，四个循环解决，主要处理问题是当最内圈不是一个矩形时，那么就要考虑两种情况：

1. 最内圈的row仅为1时
2. 最内圈的col仅为1时

row和col都为1实际上可以归类为row为1时或者col为1时的情况，所以最终只有这两个特殊情况。

代码：

```java
			if (row == tempRow) {
                for (int i = tempCol; i <= col - 1; i++) {
                    res.add(matrix[tempRow][i]);
                }
                break;
            }
            if (col == tempCol) {
                for (int i = tempRow; i <= row - 1; i++) {
                    res.add(matrix[i][col]);
                }
                break;
            }
```









## [0005. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```java
	public String longestPalindrome3(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 s[i..j] 是否是回文串
        boolean[][] dp = new boolean[len][len];
        // 初始化：所有长度为 1 的子串都是回文串
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] charArray = s.toCharArray();
        // 递推开始
        // 先枚举子串长度
        for (int L = 2; L <= len; L++) {
            // 枚举左边界，左边界的上限设置可以宽松一些
            for (int i = 0; i < len; i++) {
                // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                int j = L + i - 1;
                // 如果右边界越界，就可以退出当前循环
                if (j >= len) {
                    break;
                }
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        //这里写得很妙，这个状态转译就很到位
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                // 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }
```

动态规划：

对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。例如对于字符串 "ababa''，如果我们已经知道“bab” 是回文串，那么“ababa” 一定是回文串，这是因为它的首尾两个字母都是 “a”。

根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 P(i,j)P(i,j) 表示字符串 ss 的第 ii 到 jj 个字母组成的串（下文表示成 s[i:j]）是否为回文串：

![image-20220426112748408](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220426112748408.png)

这里的「其它情况」包含两种可能性：

- s[i, j]本身不是一个回文串；

- i > j，此时 s[i, j] 本身不合法。


那么我们就可以写出动态规划的状态转移方程：

![image-20220426112636128](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220426112636128.png)

也就是说，只有 `s[i+1:j-1]`是回文串，并且 s 的第 i 和 j 个字母相同时，s[i:j] 才会是回文串。

上文的所有讨论是建立在子串长度大于 2 的前提之上的，我们还需要考虑动态规划中的边界条件，即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，只要它的两个字母相同，它就是一个回文串。因此我们就可以写出动态规划的边界条件：

![image-20220426112619092](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220426112619092.png)

根据这个思路，我们就可以完成动态规划了，最终的答案即为所有 P(i, j) = true 中 j - i + 1（即子串长度）的最大值。**注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。**







```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    public int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            --left;
            ++right;
        }
        return right - left - 1;
    }
}
```

中心扩散法：

我们仔细观察一下方法一中的状态转移方程：

![image-20220426115732364](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220426115732364.png)

找出其中的状态转移链：

![image-20220426115744809](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220426115744809.png)

可以发现，所有的状态在转移的时候的可能性都是唯一的。也就是说，我们可以从每一种边界情况开始「扩展」，也可以得出所有的状态对应的答案。

边界情况即为子串长度为 11 或 22 的情况。我们枚举每一种边界情况，并从对应的子串开始不断地向两边扩展。如果两边的字母相同，我们就可以继续扩展，例如从 P(i+1,j-1)P(i+1,j−1) 扩展到 P(i,j)P(i,j)；如果两边的字母不同，我们就可以停止扩展，因为在这之后的子串都不能是回文串了。

聪明的读者此时应该可以发现，「边界情况」对应的子串实际上就是我们「扩展」出的回文串的「回文中心」。方法二的本质即为：我们枚举所有的「回文中心」并尝试「扩展」，直到无法扩展为止，此时的回文串长度即为此「回文中心」下的最长回文串长度。我们对所有的长度求出最大值，即可得到最终的答案。



**待扩展-马拉车Manacher's Algorithm：**











## [0031. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

```java
public void nextPermutation2(int[] nums) {
        if (nums.length <= 1) {
            return;
        }
        int start = nums[nums.length - 1];
        int continuous = 0;
        Boolean isReverse = true;
        for (int i = nums.length - 2; i >= 0; i--) {
            if (start <= nums[i]) {
                continuous++;
                start = nums[i];
            }else {
                isReverse = false;
                break;
            }
        }
        if (isReverse) {
            Arrays.sort(nums);
        }else {
            int index = nums.length - 1;
            while (true) {
                if (nums[index] > nums[nums.length - continuous - 2]) {
                    int temp = nums[nums.length - continuous - 2];
                    nums[nums.length - continuous - 2] = nums[index];
                    nums[index] = temp;
                    break;
                }
                index--;
            }
            Arrays.sort(nums, nums.length - 1 - continuous, nums.length);
        }
        return;
    }
```









## [0041. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

```java
	public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }
```

我们可以考虑将给定的数组设计成哈希表的「替代产品」。

实际上，对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 `[1,N+1]` 中。这是因为如果 `[1,N]` 都出现了，那么答案是 N+1，否则答案是 `[1,N]` 中没有出现的最小正整数。这样一来，我们将所有在 `[1,N]` 范围内的数放入哈希表，也可以得到最终的答案。而给定的数组恰好长度为 N，这让我们有了一种将数组设计成哈希表的思路：

> 我们对数组进行遍历，对于遍历到的数 x，如果它在 [1, N] 的范围内，那么就将数组中的第 x-1 个位置（注意：数组下标从 0 开始）打上「标记」。在遍历结束之后，如果所有的位置都被打上了标记，那么答案是 N+1，否则答案是最小的没有打上标记的位置加 1。
>

算法的流程如下：

1. 我们将数组中所有小于等于 0 的数修改为 N+1
2. 我们遍历数组中的每一个数 xx，它可能已经被打了标记，因此原本对应的数为 |x|，其中 || 为绝对值符号。如果 |x|∈ [1, N]，那么我们给数组中的第 |x| - 1 个位置的数添加一个负号。注意如果它已经有负号，不需要重复添加
3. 在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1

![fig1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/41_fig1.png)









## [0415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

```java
class Solution {
    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, add = 0;
        StringBuffer ans = new StringBuffer();
        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? num1.charAt(i) - '0' : 0;
            int y = j >= 0 ? num2.charAt(j) - '0' : 0;
            int result = x + y + add;
            ans.append(result % 10);
            add = result / 10;
            i--;
            j--;
        }
        // 计算完以后的答案需要翻转过来
        ans.reverse();
        return ans.toString();
    }
}

//0415. 字符串相加
public class Solution6 {
    public String addStrings(String num1, String num2) {
        if (num1.length() < num2.length()) {
            String temp = num1;
            num1 = num2;
            num2 = temp;
        }
        Deque<Character> deque = new LinkedList<>();
        int index1 = num1.length() - 1;
        int index2 = num2.length() - 1;
        Boolean isCarry = false;
        while (index2 >= 0) {
            int count = 0;
            if (isCarry) {
                count = num1.charAt(index1) - '0' + num2.charAt(index2) + 1;
            }else {
                count = num1.charAt(index1) - '0' + num2.charAt(index2);
            }
            if (count <= '9') {
                Character c = (char)(count);
                deque.addFirst(c);
                isCarry = false;
            }else {
                Character c = (char)(count - '9' + '0' - 1);
                deque.addFirst(c);
                isCarry = true;
            }
            index1--;
            index2--;
        }
        while (index1 >= 0) {
            int count = 0;
            if (isCarry) {
                count = num1.charAt(index1) + 1;
            }else {
                count = num1.charAt(index1);
            }
            if (count <= '9') {
                Character c = (char)(count);
                deque.addFirst(c);
                isCarry = false;
            }else {
                Character c = (char)(count - '9' + '0' - 1);
                deque.addFirst(c);
                isCarry = true;
            }
            index1--;
        }
        if (isCarry) {
            deque.addFirst('1');
        }
        StringBuilder sb = new StringBuilder();
        while (!deque.isEmpty()) {
            sb.append(deque.pollFirst());
        }
        return sb.toString();
    }
}
```

可以对比一下和参考答案的区别，两者思路是一样的：

定义两个指针i和j分别指向num1和num2的**==末尾==**，同时定义一个变量add维护当前是否有进位。

**参考答案是直接用的StringBuilder正常构造后reverse整个数组，而我用的是deque双向队列来实现的**。明显还是前者更好，一开始没想到。







## [0088. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

```java
//0088. 合并两个有序数组
public class Solution3 {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] res = new int[m + n];
        int count = 0;
        int index1 = 0;
        int index2 = 0;
        while (index1 < m && index2 < n) {
            if (nums1[index1] < nums2[index2]) {
                res[count] = nums1[index1];
                index1++;
            }else {
                res[count] = nums2[index2];
                index2++;
            }
            count++;
        }
        while (index1 < m) {
            res[count] = nums1[index1];
            index1++;
            count++;
        }
        while (index2 < n) {
            res[count] = nums2[index2];
            index2++;
            count++;
        }
        for (int i = 0; i < nums1.length; i++) {
            nums1[i] = res[i];
        }
    }
}

class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = 0, p2 = 0;
        int[] sorted = new int[m + n];
        int cur;
        while (p1 < m || p2 < n) {
            if (p1 == m) {
                cur = nums2[p2++];
            } else if (p2 == n) {
                cur = nums1[p1++];
            } else if (nums1[p1] < nums2[p2]) {
                cur = nums1[p1++];
            } else {
                cur = nums2[p2++];
            }
            sorted[p1 + p2 - 1] = cur;
        }
        for (int i = 0; i != m + n; ++i) {
            nums1[i] = sorted[i];
        }
    }
}
```

思路：

双指针法

![gif1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/88.%20%E5%90%88%E5%B9%B6%E4%B8%A4%E4%B8%AA%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84.png)

这个题目的要求如果再稍微苛刻一点，我觉得可以大大提升题目难度，我一开始所理解的是必须在原有nums1上做改动，最后将nums2合并进nums1，相当于全程在nums1中做改动，但是加入了新的数组用来排序则大大减小了题目难度。











## [0056. 合并区间](https://leetcode.cn/problems/merge-intervals/)

详见贪心算法中的解法







## [0076. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

```java
package stringAndArray;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

//0076. 最小覆盖子串
public class Solution8 {
    public String minWindow(String s, String t) {
        if (s.length() < t.length()) {
            return "";
        }
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            Character c = t.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        int left = 0, right = 0;
        int length = Integer.MAX_VALUE, indexL = -1, indexR = -1;
        while (right < s.length()) {
            if (map.containsKey(s.charAt(right))) {
                map.put(s.charAt(right), map.get(s.charAt(right)) - 1);
            }
            while (check(map) && left <= right) {
                if (map.containsKey(s.charAt(left))) {
                    map.put(s.charAt(left), map.get(s.charAt(left)) + 1);
                }
                if (right - left + 1 < length) {
                    length = right - left + 1;
                    indexL = left;
                    indexR = right;
                }
                left++;
            }
            right++;
        }
        if (indexL == -1) {
            return "";
        }
        return s.substring(indexL, indexR + 1);
    }

    private boolean check(Map<Character, Integer> map) {
        Iterator iter = map.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            Integer val = (Integer) entry.getValue();
            if (val > 0) {
                return false;
            }
        }
        return true;
    }
}
```

本题采用**滑动窗口**来实现：

![fig1](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/76_fig1.png)

只采用一个哈希表来完成对窗口的限制

1. 首先遍历整个字符串t，把t中所有的字符存入哈希表中，key就是其字符，value就是其字符串中所含有的字符数量。
2. 定义左右指针，当哈希表中所有元素的value都大于0时，**移动右指针去扩大窗口**，因为说明此说滑动窗口中并没有包含整个字符串t所有元素；当哈希表中所有元素的value都小于0时，此时窗口中已经包含了字符串t中的所有元素，**移动左指针去缩小窗口**。
3. 同步更新哈希表中元素的value值。







## [0165. 比较版本号](https://leetcode.cn/problems/compare-version-numbers/)

```java
package stringAndArray;
//0165. 比较版本号
public class Solution9 {
    public int compareVersion(String version1, String version2) {
        int indexV1 = 0, indexV2 = 0;
        while (indexV1 < version1.length() && indexV2 < version2.length()) {
            StringBuilder sb1 = new StringBuilder();
            StringBuilder sb2 = new StringBuilder();
            while (indexV1 < version1.length() && version1.charAt(indexV1) != '.') {
                //这一步是清除前导0用的
                if (sb1.length() == 0 && version1.charAt(indexV1) == '0') {
                    indexV1++;
                    continue;
                }
                sb1.append(version1.charAt(indexV1));
                indexV1++;
            }
            while (indexV2 < version2.length() && version2.charAt(indexV2) != '.') {
                if (sb2.length() == 0 && version2.charAt(indexV2) == '0') {
                    indexV2++;
                    continue;
                }
                sb2.append(version2.charAt(indexV2));
                indexV2++;
            }
            //当版本号是".0."的情况时，由于要过滤前导0，所以sb中此时是空的，那么就需要做一个判断sb.length()是否大于0，如果是0则手动给i赋值为0
            Integer i1 = 0;
            if (sb1.length() > 0) {
                i1 = Integer.valueOf(sb1.toString());
            }
            Integer i2 = 0;
            if (sb2.length() > 0) {
                i2 = Integer.valueOf(sb2.toString());
            }
            if (i1 > i2) {
                return 1;
            }
            if (i2 > i1) {
                return -1;
            }
            indexV1++;
            indexV2++;
        }
        if (indexV1 >= version1.length()) {
            while (indexV2 < version2.length()) {
                StringBuilder sb = new StringBuilder();
                while (indexV2 < version2.length() && version2.charAt(indexV2) != '.') {
                    if (sb.length() == 0 && version2.charAt(indexV2) == '0') {
                        indexV2++;
                        continue;
                    }
                    sb.append(version2.charAt(indexV2));
                    indexV2++;
                }
                //如果sb中能塞入元素，说明一定有非0元素，而本题是没有负数的，所以只要有元素进sb，就一定是大于0的
                if (sb.length() > 0) {
                    return -1;
                }
                indexV2++;
            }
        }
        if (indexV2 >= version2.length()) {
            while (indexV1 < version1.length()) {
                StringBuilder sb = new StringBuilder();
                while (indexV1 < version1.length() && version1.charAt(indexV1) != '.') {
                    if (sb.length() == 0 && version1.charAt(indexV1) == '0') {
                        indexV1++;
                        continue;
                    }
                    sb.append(version1.charAt(indexV1));
                    indexV1++;
                }
                if (sb.length() > 0) {
                    return 1;
                }
                indexV1++;
            }
        }
        return 0;
    }
}
```

本题的目的是要去对比版本号大小，题目中可能出现的问题：

1. 前导0：由于前导0最后全部是要删除的，所以需要有一个判断在这里面，并且有一个例子很特殊就是".0."，因为我是用StringBuilder来构造'.'与'.'之间的版本号，而又有要过滤前导0的情况，所以".0."的情况最后会导致StringBuilder里是空的。
2. 两版本号的结构不同：两个版本号可能由不同数量的'.'组成，这就导致了需要有提前结束循环的条件，也就是有字符串verison1和version2两个条件。

思路：

1. 首先是一个循环，保证version1和version2任意一个在遍历完成后及时结束循环：

   `while (indexV1 < version1.length() && indexV2 < version2.length())`

   这个循环每次都重置一次StringBuilder sb1和sb2，并且要在最后给indexV1和indexV2++，因为下面循环出来时遍历到的一定是'.'(或者是遍历完了)，所以要把'.'跳过

2. 第二个循环：

   ```java
   			while (indexV1 < version1.length() && version1.charAt(indexV1) != '.') {
                   //这一步是清除前导0用的
                   if (sb1.length() == 0 && version1.charAt(indexV1) == '0') {
                       indexV1++;
                       continue;
                   }
                   sb1.append(version1.charAt(indexV1));
                   indexV1++;
               }			
   ```

   `sb.length() == 0`时又遍历到'0'，这个'0'就不入sb





## [0704. 二分查找](https://leetcode.cn/problems/binary-search/)

```java
//0704. 二分查找
public class Solution12 {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }
}
```

一个经典的二分查找结构



## [0162. 寻找峰值(二分查找)](https://leetcode.cn/problems/find-peak-element/)

```java
	public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        for (; left < right; ) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
```

- 标签：二分查找

- 过程：
  - 首先要注意题目条件，在题目描述中出现了 nums[-1] = nums[n] = -∞，这就代表着 只要数组中存在一个元素比相邻元素大，那么沿着它一定可以找到一个峰值
  - 根据上述结论，我们就可以使用二分查找找到峰值
  - 查找时，左指针 l，右指针 r，以其保持左右顺序为循环条件
  - 根据左右指针计算中间位置 m，并比较 m 与 m+1 的值，如果 m 较大，则左侧存在峰值，r = m，如果 m + 1 较大，则右侧存在峰值，l = m + 1





## [0004. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }

        int m = nums1.length;
        int n = nums2.length;
        int left = 0, right = m;
        // median1：前一部分的最大值
        // median2：后一部分的最小值
        int median1 = 0, median2 = 0;

        while (left <= right) {
            // 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
            // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            int i = (left + right) / 2;
            int j = (m + n + 1) / 2 - i;

            // nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            int nums_im1 = (i == 0 ? Integer.MIN_VALUE : nums1[i - 1]);
            int nums_i = (i == m ? Integer.MAX_VALUE : nums1[i]);
            int nums_jm1 = (j == 0 ? Integer.MIN_VALUE : nums2[j - 1]);
            int nums_j = (j == n ? Integer.MAX_VALUE : nums2[j]);

            if (nums_im1 <= nums_j) {
                median1 = Math.max(nums_im1, nums_jm1);
                median2 = Math.min(nums_i, nums_j);
                left = i + 1;
            } else {
                right = i - 1;
            }
        }

        return (m + n) % 2 == 0 ? (median1 + median2) / 2.0 : median1;
    }
```

https://leetcode.cn/problems/median-of-two-sorted-arrays/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-s-114/

二分查找法：

首先将nums1和nums2两个集合划分成left_A，left_B和right_A，right_B(**==注意本解析中的i和j指的是右半部分的第一个元素，而不是左半部分的最后一个元素==**)：

```
          left_part          |         right_part
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]
```

当A和B的总长度为偶数时，需要去找到以下条件：

- len(left_part) = len(right_part)
- max(left_part) ≤ min(right_part)

可以得出中位数为：
$$
median=\frac{max(left_{part})+min(right_{part})}{2}
$$
当A和B的总长度为奇数时，需要找到以下条件：

- len(left_part) = len(right_part) + 1
- max(left_part) ≤ min(right_part)

可以得出中位数为：
$$
median=max(left_{part})
$$
要确保以上条件需要保证：

`i + j = m - i + n - j + 1 -> j = (m + n + 1) / 2 - i`

![image-20220801095803156](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220801095803156.png)

![image-20220801095148527](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220801095148527.png)

边界条件：

1. A[-1] = B[-1] = -∞
2. A[m] = B[N] = ∞





## [0010. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)

```java
public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }
```

**选择从右往左扫描：**

- 星号的前面肯定是一个符号，星号也只能影响这一个字符

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/5e7b1748039a2a779d7378bebc4926ef3e584e88cc22b67f3a4e18c0590bcc55-image.png)

- s、p 串是否匹配，取决于：最右端是否匹配、剩余的子串是否匹配。
- 只是最右端可能是特殊符号，需要分情况讨论而已。

**通用地表示子问题**

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/e1bcac2ad07a3a5c959bf0fe5c8ceea9bbd033c3066e7ec7f384aedd98cd95aa-image.png)

**当s[i-1]和p[j-1]是匹配的**

最右端的字符匹配，则此时考虑剩余子串是否匹配

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/f817caaa40b0c39fc3ddabfa1383a8218ab364b8e49b30e5ce85cb30a3cdc503-image.png)

**s[i-1]和p[j-1]不匹配**

- 右端不匹配，还不能判死刑——可能是 p[j-1]p[j−1] 为星号造成的不匹配，星号不是真实字符，它不匹配不算数。
- 如果 p[j-1]p[j−1] 不是星号，那就真的不匹配了。

![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/fe763378879a0a52e9f17171e3bc1db18cfc83bf59f14efcd31ec9edb37adfac-image.png)

**`p[j−1]=='*'`，且 s[i−1] 和 p[j−2] 匹配**

- p[j-1]是星号的话，可以让p[j-2]在p串中：

  - 直接排除
  - 出现一次
  - 出现>=2次

  ![image.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/a1cc0caf806f7d7f5419d820e0e7be7a364c96656a98ca4d7f351661d6a62aa6-image.png)











## [0011. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

```java
public int maxArea(int[] height) {
        if (height.length < 2) {
            return 0;
        }
        int left = 0, right = height.length - 1;
        int max = (right - left) * Math.min(height[left], height[right]);
        while (left < right) {
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
            int temp = (right - left) * Math.min(height[left], height[right]);
            max = max >= temp ? max : temp;
        }
        return max;
    }
```

双指针

在每个状态下，无论长板或短板向中间收窄一格，都会导致水槽 **底边宽度** -1 变短：

- 若向内 移动短板 ，水槽的短板 min(h[i], h[j])min(h[i],h[j]) 可能变大，因此下个水槽的面积 可能增大 。
- 若向内 移动长板 ，水槽的短板 min(h[i], h[j])min(h[i],h[j]) 不变或变小，因此下个水槽的面积 一定变小 。

因此，初始化双指针分列水槽左右两端，循环每轮将短板向内移动一格，并更新面积最大值，直到两指针相遇时跳出；即可获得最大面积。









## [0075. 颜色分类](https://leetcode.cn/problems/sort-colors/)

```java
public void sortColors(int[] nums) {
    int red = 0, white = 0;
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] == 2) {
            continue;
        } else if (nums[i] == 1) {
            int temp = nums[i];
            nums[i] = nums[white];
            nums[white] = temp;
            white++;
        } else if (nums[i] == 0) {
            int temp = nums[i];
            nums[i] = nums[red];
            nums[red] = temp;
            if (red < white) {
                temp = nums[i];
                nums[i] = nums[white];
                nums[white] = temp;
            }
            red++;
            white++;
        }
    }
}
```

双指针

用指针red来交换0，用指针white来交换1，初始都在index=0的位置。

- 如果遍历到`nums[i]==2`那么直接跳过本次循环
- 如果遍历到`nums[i]==1`先互换nums[i]和nums[white]的位置，然后让white++
- 如果遍历到`nums[i]==0`先互换nums[i]和nums[red]的位置，如果`red<white`证明前面已经有白球已经被换过了，此时单纯的互换nums[i]和nums[red]一定会把一个1换到尾部（此时nums[i]是1），所以要再把nums[i]和nums[white]做一次互换，把2换到尾部。







## [0079. 单词搜索](https://leetcode.cn/problems/word-search/)

```java
public boolean exist(char[][] board, String word) {
    for (int i = 0; i < board.length; i++) {
        for (int j = 0; j < board[0].length; j++) {
            if (board[i][j] == word.charAt(0)) {
                int[][] isUsed = new int[board.length][board[0].length];
                if (useDFS(isUsed, board, 0, word, i , j)) {
                    return true;
                }
            }
        }
    }
    return false;
}
private boolean useDFS(int[][] isUsed, char[][] board, int wordIndex, String word, int i, int j) {
    //当遍历到的位置越界即返回
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
        return false;
    }
    如果遍历到的位置已经被遍历过则返回
    if (isUsed[i][j] != 0) {
        return false;
    }
    if (board[i][j] == word.charAt(wordIndex)) {
        isUsed[i][j] = 1;
        wordIndex++;
        if (wordIndex == word.length()) {
            return true;
        }
        //深度优先的四个方向，按照上右下左的顺序依次去遍历
        if (useDFS(isUsed, board, wordIndex, word, i - 1, j)) {
            return true;
        }
        if (useDFS(isUsed, board, wordIndex, word, i, j + 1)) {
            return true;
        }
        if (useDFS(isUsed, board, wordIndex, word, i + 1, j)) {
            return true;
        }
        if (useDFS(isUsed, board, wordIndex, word, i, j - 1)) {
            return true;
        }
        isUsed[i][j] = 0;//这里回溯
    }
    return false;
}
```

本题是一个搜索过程，方向有四个方向，上下左右，所有可以用**DFS**和回溯来完成题目解答





## [0136. 只出现一次的数字---异或运算](https://leetcode.cn/problems/single-number/)

```java
public int singleNumber(int[] nums) {
    int res = nums[0];
    for (int i = 1; i < nums.length; i++) {
        res = res ^ nums[i];
    }
    return res;
}
```

位运算：

使用异或运算⊕，异或运算的特性：

- 任何数与0做异或运算，结果仍为原来的数，即a⊕0=a
- 任何数和其自身做异或运算，结果是0，即a⊕a=a
- 异或运算满足交换律和结合律，即a⊕b⊕a=b⊕a⊕a=b⊕(a⊕a)=b⊕0=b







## [0169. 多数元素](https://leetcode.cn/problems/majority-element/)

```java
public int majorityElement(int[] nums) {
    int candidate = nums[0];
    int count = 1;
    for (int i = 1; i< nums.length; i++) {
        if (count == 0) {
            candidate = nums[i];
            count = 1;
            continue;
        }
        if (candidate == nums[i]) {
            count++;
        } else {
            count--;
        }
    }
    return candidate;
}
```

由于题目要求尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题

而已知解法有排序法(但是排序首先空间复杂度就上升到了O^2)，或者使用哈希表来进行解题(但是空间复杂度就不是O1了)

**摩尔投票法**

多数元素一定只存在一个，并且数量大于n/2，那么就意味着，多数元素与剩下的全部元素的数量差最小为1(即>=1)。

投票法是遇到相同的则`票数+1`，遇到不同的就`票数-1`

将多数元素与剩余元素两两抵消，到最后肯定还剩余至少1个多数元素













## [0238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

```java
public int[] productExceptSelf(int[] nums) {
    int[] left = new int[nums.length];
    int[] right = new int[nums.length];
    left[0] = nums[0];
    right[right.length - 1] = nums[nums.length - 1];
    for (int i = 1; i < nums.length; i++) {
        left[i] = left[i - 1] * nums[i];
    }
    for (int i = right.length - 2; i >= 0; i--) {
        right[i] = right[i + 1] * nums[i];
    }
    for (int i = 0; i < nums.length; i++) {
        nums[i] = (i == 0 ? 1 : left[i - 1]) * (i == nums.length - 1 ? 1 : right[i + 1]);
    }
    return nums;
}
```

创建两个数组，分别记录元素i左边元素的乘积以及i右边元素的乘积

- 左边：

  ```java
  for (int i = 1; i < nums.length; i++) {
          left[i] = left[i - 1] * nums[i];
      }
  ```

- 右边：

  ```java
  for (int i = right.length - 2; i >= 0; i--) {
          right[i] = right[i + 1] * nums[i];
      }
  ```



本题进阶要求使用常数空间复杂度来完成：

本题说输出数组不算额外空间（感觉这题非常垃圾）

把输出数组首先更新为left存放左边元素乘积的数组，而right数组可以动态生成，因为nums输入数组还在，从右边再往左边遍历一次即可。

public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];

```java
    // answer[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
    answer[0] = 1;
    for (int i = 1; i < length; i++) {
        answer[i] = nums[i - 1] * answer[i - 1];
    }

    // R 为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 R = 1
    int R = 1;
    for (int i = length - 1; i >= 0; i--) {
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        answer[i] = answer[i] * R;
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i];
    }
    return answer;
}
```









## [0253. 会议室 II](https://leetcode.cn/problems/meeting-rooms-ii/)

```java
public int minMeetingRooms(int[][] intervals) {
    Arrays.sort(intervals, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    PriorityQueue<Integer> pq = new PriorityQueue<>(
            intervals.length,
            new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return o1 - o2;
                }
            }
    );
    pq.add(intervals[0][1]);
    int max = 1;
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] >= pq.peek()) {
            pq.poll();
        }
        pq.add(intervals[i][1]);
        max = Math.max(max, pq.size());
    }
    return max;
}
```

本题思路就是先对所有会议时间进行排序，根据会议的开始时间来从小到大排序，这样等效保证一个连续的时间线

由于会议开始时间的顺序已经被排序好了，那么就是要看会议的结束时间，下一场会议的开始时间是否是在当前会议的结束时间之后，如果是的话，那么就可以不另外再加一个会议室，反之亦然。我们可以把会议室想象成一个会议池，没有结束的会议都会被扔进池子里，结束时离开池子。这就可以使用优先队列，根据会议的结束时间来排序，每次有新会议开始的时候，把会议结束时间早于当前会议的开始时间的会议全部出队即可。然后持续关注优先队列的大小，即可得到最大会议室值。







## [0301. 删除无效的括号](https://leetcode.cn/problems/remove-invalid-parentheses/)

```java
Set<String> set = new HashSet<>();
int max = 0, maxL = 0;//max是当前合法组合的最大数组长度，maxL是字符串s合法子串的最多左括号数量
public List<String> removeInvalidParentheses(String s) {
    int l = 0, r= 0;
    for (char c : s.toCharArray()) {
        if (c == '(') l++;
        if (c == ')') r++;
    }
    maxL = Math.max(l, r);
    dfs(s, new String(), 0, 0, 0);
    return new ArrayList<>(set);
}

private void dfs(String s, String constructS, int index, int length, int balance) {
    if (balance == 0 && length >= max) {
        if (length > max) set.clear();
        max = length;
        set.add(constructS);
    }
    if (index >= s.length() || balance < 0 || balance > maxL) {
        return;
    }
    int tempBalance;
    if (s.charAt(index) == '(') {
        tempBalance = 1;
    } else if (s.charAt(index) == ')') {
        tempBalance = -1;
    } else {
        tempBalance = 0;
    }
    dfs(s, constructS + s.charAt(index), index + 1, length + 1, balance + tempBalance);
    dfs(s, constructS, index + 1, length, balance);
}
```

我们利用一个变量balance来判断子字符串是否合法，一个左括号的价值是+1，一个右括号的价值是-1，其它字符的价值是0。

- 当balance<0时，说明某一个前缀字符串的起始是右括号，因此非法
- 本题的递归思路就是，遍历输入字符串s，对每个字符都可以有选与不选两个操作（DFS）







## [0030. 串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)

```java
public List<Integer> findSubstring(String s, String[] words) {
    Map<String, Integer> map = new HashMap<>();
    int subLength = 0;
    int wordLength = words[0].length();
    for (String word : words) {
        subLength += wordLength;
        map.put(word, map.getOrDefault(word, 0) + 1);
    }
    List<Integer> res = new ArrayList<>();
    for (int left = 0; left < s.length(); left++) {
        if (s.length() - left < subLength) {
            break;
        }
        Map<String, Integer> curMap = new HashMap<>(map);
        Boolean isSub = true;
        for (int right = left; right < left + subLength; right += wordLength) {
            String subString = s.substring(right, right + wordLength);
            if (!curMap.containsKey(subString)) {
                isSub = false;
                break;
            }
            if (curMap.get(subString) - 1 == 0) {
                curMap.remove(subString);
            } else {
                curMap.put(subString, curMap.get(subString) - 1);
            }
        }
        if (isSub) {
            res.add(left);
        }
    }
    return res;
}
```

本题首先创建一个哈希表来存放words中所有的单词，在记录完words一共存有的单词数量后，计算得到subString的长度后开始对字符串s进行遍历

1. 每轮循环都创建一个新的哈希表，把初始哈希表的值copy过去
2. 利用左右指针，左指针每次移动一格，右指针每次从左指针位置开始，每次移动word.length()的格数，判断s.substring(right, right + wordLength)是否存在于哈希表中，如果不存在，说明从left位置开始不存在串联子串。
3. 如果该单词存在于哈希表中，则把对应的值-1，如果-1后得到的value为0，那么就把该元素从map中移除，这样就可以用map.containKey(word)来进行后续判断，如果该子串是串联子串，最后map应该是空的。





## [01455. 检查单词是否为句中其他单词的前缀](https://leetcode.cn/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/)

```java
public int isPrefixOfWord(String sentence, String searchWord) {
    if (sentence.length() == 0 || searchWord.length() == 0) {
        return -1;
    }
    int count = 1;
    int index = 0;
    Boolean isMatch = true;
    for (int i = 0; i < sentence.length(); i++) {
        if (sentence.charAt(i) == ' ') {
            count++;
            index = 0;
            isMatch = true;
            continue;
        }
        if (!isMatch) {
            continue;
        }
        if (sentence.charAt(i) == searchWord.charAt(index)) {
            if (index == searchWord.length() - 1) {
                return count;
            }
            index++;
        } else {
            isMatch = false;
        }
    }
    return -1;
}
```

题目已经写好了构造函数，限制了开头字母不能是空格，所以说就可以放心大胆设置下标为0开始即就是第一个单词的开始。

定义count用来表示当前遍历到的是第几个单词，index代表searchWord检索词当前检索到的位置，isMatch表示在sentence当前单词中，是否已经出现与检索词不匹配的字母。

1. 每次遍历到`' '`空格，就需要把count++，并把index和isMatch重置
2. 具体看代码吧，很清楚







# Linked List

## [0025. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

```java
	public ListNode reverseKGroup(ListNode head, int k) {
        if (k == 1) {
            return head;
        }
        Deque<ListNode> deque = new LinkedList<>();
        ListNode dummy = new ListNode();
        ListNode tempEnd = dummy;
        ListNode cur = head;
        int count = 0;
        while (cur != null) {
            deque.addLast(cur);
            cur = cur.next;
            count++;
            if (count == k) {
                count = 0;
                while (!deque.isEmpty()) {
                    tempEnd.next = deque.pollLast();
                    tempEnd = tempEnd.next;
                    if (cur == null && deque.isEmpty()) {
                        tempEnd.next = null;
                    }
                }
            }
        }
        if (!deque.isEmpty()) {
            tempEnd.next = deque.pollFirst();
        }
        return dummy.next;
    }
```

思路：

新建一个虚拟头结点，如果不是虚拟头结点的链表处理就要单独判断每一个循环的头元素，会很麻烦

1. 本题主要用Deque双向队列来解决问题

2. 第一层结构是对head这个链表进行遍历，首先让`cur = head`，然后去对cur遍历，`while (cur != null)`

3. 循环内部，每次都让`cur = cur.next`，每次都往deque的尾部添加新元素，当循环每累积4次就开始排空deque，这个过程deque当作堆栈来使用，这样的话就能起到一个反转效果：

   ```java
   				while (!deque.isEmpty()) {
                       tempEnd.next = deque.pollLast();
                       tempEnd = tempEnd.next;
                       if (cur == null && deque.isEmpty()) {
                           tempEnd.next = null;
                       }
                   }
   ```

   这里解释一下`if (cur == null && deque.isEmpty())`，这个就是对于最后一组反转链表的操作，它的末尾元素不会有新的`tempEnd.next = deque.pollLast()`操作了，那么就会形成一个循环链表，必须给它`tempEnd.next = null`。

4. 如果是第一层循环先结束，就是说最后有剩余节点不需要做反转操作，那么直接判断deque是否为空，不为空，将`tempEnd.next = deque的头元素`即可。







## [0160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)*

```java
//0160. 相交链表
public class GetIntersectionNode {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Set<ListNode> setA = new HashSet<>();
        Set<ListNode> setB = new HashSet<>();
        while (headA != null && headB != null) {
            setA.add(headA);
            setB.add(headB);
            if (setA.contains(headB)) {
                return headB;
            }
            if (setB.contains(headA)) {
                return headA;
            }
            headA = headA.next;
            headB = headB.next;
        }
        while (headA != null) {
            if (setB.contains(headA)) {
                return headA;
            }
            headA = headA.next;
        }
        while (headB != null) {
            if (setA.contains(headB)) {
                return headB;
            }
            headB = headB.next;
        }
        return null;
    }
}
```

没啥特别的思路，就是利用遍历走完全部的节点，利用两个hashset来记录A，B两个链表所经过的全部节点，如果：

```java
			if (setA.contains(headB)) {
                return headB;
            }
            if (setB.contains(headA)) {
                return headA;
            }
```

以下写法更好：

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Set<ListNode> visited = new HashSet<ListNode>();
        ListNode temp = headA;
        while (temp != null) {
            visited.add(temp);
            temp = temp.next;
        }
        temp = headB;
        while (temp != null) {
            if (visited.contains(temp)) {
                return temp;
            }
            temp = temp.next;
        }
        return null;
    }

```



解法二-双指针：

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode pA = headA;
        ListNode pB = headB;
        while (true) {
            if (pA == pB) {
                return pA;
            }
            if (pA == null && pB == null) {
                return null;
            }
            if (pA == null) {
                pA = headB;
            }else {
                pA = pA.next;
            }
            if (pB == null) {
                pB = headA;
            }else {
                pB = pB.next;
            }
        }
    }
```

空间复杂度 O(1)O(1) 时间复杂度为 O(n)O(n)

这里使用图解的方式，解释比较巧妙的一种实现。

根据题目意思
如果两个链表相交，那么相交点之后的长度是相同的

我们需要做的事情是，让两个链表从同距离末尾同等距离的位置开始遍历。这个位置只能是较短链表的头结点位置。
为此，我们必须消除两个链表的长度差

指针 pA 指向 A 链表，指针 pB 指向 B 链表，依次往后遍历
如果 pA 到了末尾，则 pA = headB 继续遍历
如果 pB 到了末尾，则 pB = headA 继续遍历
比较长的链表指针指向较短链表head时，长度差就消除了
如此，只需要将最短链表遍历两次即可找到位置
听着可能有点绕，看图最直观，链表的题目最适合看图了

![相交链表.png](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/e86e947c8b87ac723b9c858cd3834f9a93bcc6c5e884e41117ab803d205ef662-%E7%9B%B8%E4%BA%A4%E9%93%BE%E8%A1%A8.png)

代码优化：

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) return null;
    ListNode pA = headA, pB = headB;
    while (pA != pB) {
        pA = pA == null ? headB : pA.next;
        pB = pB == null ? headA : pB.next;
    }
    return pA;
}
```





## [0023. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

```java
	public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = new ListNode();
        ListNode tempNode = dummy;
        Boolean[] isNull = new Boolean[lists.length];
        Arrays.fill(isNull, false);
        int countNull = 0;
        while (true) {
            int min = 10001;
            int index = 0;
            for (int i = 0; i < lists.length; i++) {
                if (isNull[i] == true) {
                    continue;
                }
                if (lists[i] == null) {
                    isNull[i] = true;
                    countNull++;
                    continue;
                }
                min = min < lists[i].val ? min : lists[i].val;
                index = min < lists[i].val ? index : i;
            }
            if (countNull == lists.length) {
                break;
            }
            lists[index] = lists[index].next;
            ListNode newNode = new ListNode(min);
            tempNode.next = newNode;
            tempNode = tempNode.next;
        }
        return dummy.next;
    }
```

思路：

反复对lists中的元素进行遍历，比较每次遍历中最小值（所有元素都是单调递增的链表），记录当前遍历中最小的那个元素的index，将这个元素的链表后移`lists[index] = lists[index].next`，然后将这个最小元素拼接进新链表：

```java
ListNode newNode = new ListNode(min);
tempNode.next = newNode;
tempNode = tempNode.next;
```

循环退出的判断条件为lists中所有元素均为null。我的实现方法是，再定义一个isNull的Boolean数组（初始化全为false），每次遍历到一个链表为null时，就把`isNull[i]`变为true，并把countNull++（当`countNull == lists.length`时就是退出循环的条件了）。





```java
public ListNode mergeKLists2(ListNode[] lists) {
    Queue<ListNode> pq = new PriorityQueue<>((v1, v2) -> v1.val - v2.val);
    for (ListNode node: lists) {
        if (node != null) {
            pq.offer(node);
        }
    }

    ListNode dummyHead = new ListNode(0);
    ListNode tail = dummyHead;
    while (!pq.isEmpty()) {
        ListNode minNode = pq.poll();
        tail.next = minNode;
        tail = minNode;
        if (minNode.next != null) {
            pq.offer(minNode.next);
        }
    }

    return dummyHead.next;
}
```

思路：

使用小根堆对 1 进行优化，每次 O(logK)*O*(logK) 比较 K个指针求 min, 时间复杂度：O(NlogK)*O*(NlogK)

利用PriorityQueue来实现队列顺序，











## [0143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

```java
//0143. 重排链表
public class ReorderList {
    public void reorderList(ListNode head) {
        ListNode temp = head;
        while (true) {
            ListNode next = temp.next;
            ListNode newHead = temp.next;
            if (temp.next == null) {
                break;
            }
            if (temp.next.next == null) {
                break;
            }
            while (true) {
                if (next.next.next == null) {
                    break;
                }
                next = next.next;
            }
            ListNode end = next.next;
            temp.next = end;
            next.next = null;
            end.next = newHead;
            temp = newHead;
        }
    }
}
```

思路：

每个循环的逻辑都是：

1. 先找到倒数第二个节点node，然后利用node.next得到end节点
2. node.next = null，切断连接
3. 把head连到end节点
4. 把end节点连到head.next







```java
class Solution {
    public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }
        ListNode mid = middleNode(head);
        ListNode l1 = head;
        ListNode l2 = mid.next;
        mid.next = null;
        l2 = reverseList(l2);
        mergeList(l1, l2);
    }

    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    public void mergeList(ListNode l1, ListNode l2) {
        ListNode l1_tmp;
        ListNode l2_tmp;
        while (l1 != null && l2 != null) {
            l1_tmp = l1.next;
            l2_tmp = l2.next;

            l1.next = l2;
            l1 = l1_tmp;

            l2.next = l1;
            l2 = l2_tmp;
        }
    }
}
```

方法二：寻找链表中点 + 链表逆序 + 合并链表
注意到目标链表即为将原链表的左半端和反转后的右半端合并后的结果。

这样我们的任务即可划分为三步：

1. 找到原链表的中点（参考「876. 链表的中间结点」）。

   我们可以使用快慢指针来 O(N)O(N) 地找到链表的中间节点。

2. 将原链表的右半端反转（参考「206. 反转链表」）。

   我们可以使用迭代法实现链表的反转。

3. 将原链表的两端合并。

   因为两链表长度相差不超过 11，因此直接合并即可。

















## [0876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

```java
//0876. 链表的中间结点
public class MiddleNode {
    public ListNode middleNode(ListNode head) {
        ListNode mid = head;
        ListNode end = head;
        while (true) {
            if (end.next == null) {
                return mid;
            }
            if (end.next.next == null) {
                return mid.next;
            }
            end = end.next.next;
            mid = mid.next;
        }
    }
}
```

找中间节点的方法就是用两个指针，指针1用来定位mid的位置，指针2用来定位end的位置，mid每次动1，end每次动2。根据这个规律当end移动到末尾时即：

1. end.next == null时，这个条件就意味着链表长度为奇数，直接返回mid此时位置。
2. end.next != null，end.next.next == null，这个条件意味着链表长度为偶数，此时要返回mid.next。







## [0021. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```java
//0021. 合并两个有序链表
public class Solution2 {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        ListNode head;
        ListNode pre1 = new ListNode();
        ListNode pre2 = new ListNode();
        head = list1.val >= list2.val ? list2 : list1;
        while (true) {
            if (list1.val >= list2.val) {
                pre1.next = list2;
                while (list2 != null && list1.val >= list2.val) {
                    pre2 = list2;
                    list2 = list2.next;
                }
                pre2.next = list1;
            }
            if (list1.next == null || list2 == null) {
                break;
            }
            pre1 = list1;
            list1 = list1.next;
        }
        if (list2 != null) {
            list1.next = list2;
        }
        return head;
    }
}
```

思路：

首先先定义新链表的头结点，我的新链表构造思路是(假设有链表A和链表B)：

1. 首先定义两个结点，分别为pre1和pre2，分别代表A和B当前结点的前一个结点。

2. 如果链表B当前结点值小于或者等于链表A结点值，把A的当前结点的前一个结点链接到链表B当前结点

3. 如果链表B当前结点值小于或者等于链表A结点值，则全部放到链表当前结点前

   ```java
   if (list1.val >= list2.val) {
                   pre1.next = list2;
                   while (list2 != null && list1.val >= list2.val) {
                       pre2 = list2;
                       list2 = list2.next;
                   }
                   pre2.next = list1;
               }
   ```

4. 重复2操作，但如果链表B当前结点值大于链表A结点值

   ```java
   			pre1 = list1;
               list1 = list1.next;
   ```

根据以上逻辑，可以这样来判断头结点应该选链表A还是链表B：`head = list1.val >= list2.val ? list2 : list1`



代码优化：

```java

class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);

        ListNode prev = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 == null ? l2 : l1;

        return prehead.next;
    }
}

```







```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        } else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
```

思路2-递归：

我们可以如下递归地定义两个链表里的 `merge` 操作（忽略边界情况，比如空链表等）：

![image-20220505104618417](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220505104618417.png)

也就是说，两个链表头部值较小的一个节点与剩下元素的 `merge` 操作结果合并。

我们直接将以上递归过程建模，同时需要考虑边界情况。

如果 l1 或者 l2 一开始就是空链表 ，那么没有任何操作需要合并，所以我们只需要返回非空链表。否则，我们要判断 l1 和 l2 哪一个链表的头节点的值更小，然后递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束。







## [0092. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

```java
//0092. 反转链表 II
public class Solution4 {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode start = new ListNode();
        if (left == 1) {
            start = dummy;
        }
        Deque<ListNode> deque = new LinkedList<>();
        for (int i = 1; i <= right; i++) {
            if (i < left) {
                if (i == left - 1) {
                    start = head;
                }
                head = head.next;
            }else {
                deque.addLast(head);
                head = head.next;
            }
        }
        ListNode end = head;//此时的head就是right + 1的结点
        ListNode temp = start;
        while (!deque.isEmpty()) {
            temp.next = deque.pollLast();
            temp = temp.next;
        }
        temp.next = end;
        return dummy.next;
    }
}
```

思路：

首先要判断一种特殊情况，就是当left == 1的时候，这时候head头结点直接相当于也会有改动，为了解决这个问题，我提前定义了一个dummy虚拟头结点，连接到头结点。start结点就是记录开始反转位置前的结点(left - 1)，end结点就是记录反转结束位置后一个结点(right + 1)。

1. 如果left == 1，则start就为dummy
2. 如果left > 1，则start正常在遍历中遇到i == left - 1时赋值

最后出循环的时候，把此时结点赋值给end，出循环的时候就是right + 1的结点









## [0141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

```java
//0141. 环形链表
public class Solution5 {
    public boolean hasCycle(ListNode head) {
        Boolean isLoop = false;
        if (head == null) {
            return isLoop;
        }
        ListNode left = head;
        ListNode right = head;
        while (right.next != null && right.next.next != null) {
            left = left.next;
            right = right.next.next;
            if (left == right) {
                isLoop = true;
                break;
            }
        }
        return isLoop;
    }
}
```

定义快慢两个指针，一个每次+1，一个每次+2，如果两个会相遇就会出现`left == right`的情况。













## [0142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

```java
//0142. 环形链表 II
public class Solution6 {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        ListNode intersect = null;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                intersect = slow;
                break;
            }
        }
        fast = intersect;
        slow = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

详见ipad笔记

![IMG_0766](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/IMG_0766.png)

![image-20220714095428760](/Users/qinyang/markdown/LeetCode/图片/image-20220714095428760.png)



## [0148. 排序链表](https://leetcode.cn/problems/sort-list/)

```java
	public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }
    public ListNode sortList(ListNode head, ListNode tail){
        if (head == null) {
            return head;
        }
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode slow = head, fast = head;
        //while (fast.next != null)
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }
        ListNode mid = slow;
        ListNode list1 = sortList(head, mid);
        ListNode list2 = sortList(mid, tail);
        ListNode sorted = merge(list1, list2);
        return sorted;
    }
    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            //别忘了让temp往后走
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummy.next;
    }
```

递归：

**自顶向下归并排序**

对链表自顶向下归并排序的过程如下。

1. 找到链表的中点，以中点为分界，将链表拆分成两个子链表。寻找链表的中点可以使用快慢指针的做法，快指针每次移动 2步，慢指针每次移动 1步，当快指针到达链表末尾时，慢指针指向的链表节点即为链表的中点。

2. 对两个子链表分别排序。

3. 将两个排序后的子链表合并，得到完整的排序后的链表。可以使用「21. 合并两个有序链表」的做法，将两个有序的子链表进行合并。


实际上就是由3个非常基础的链表操作组合完成。第一步就是一直递归去寻找每一个链表的中点，依次下去最后一定会定位到单独的一个节点，然后从最底层的节点开始进行排序，再变成排序好的子链表，最后再对子链表进行排序，最后得到排序好的链表。







## [0002. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

```java
package list;
//0002. 两数相加
public class Solution8 {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head1 = l1, head2 = l2;
        ListNode ansDummy = new ListNode();//虚拟头结点可以让头结点的处理不那么复杂
        ListNode cur = ansDummy;
        Boolean carry = false;//判断是否进位
        while (l1 != null && l2 != null) {//任意一个链表先遍历完就直接退出循环，之后单独对未遍历完链表做操作
            ListNode temp = new ListNode();
            if (carry) {//因为是两数相加，所以进位最多只会+1
                temp.val = l1.val + l2.val + 1;
            } else {
                temp.val = l1.val + l2.val;
            }
            if (temp.val >= 10) {//如果此位和大于等于10那么就要进位
                temp.val = temp.val - 10;
                carry = true;
            } else {
                carry = false;
            }
            l1 = l1.next;
            l2 = l2.next;
            cur.next = temp;
            cur = cur.next;
        }
        //以下两个while的operation一模一样，如果两个链表不一样长，则一定有一个会先遍历完，然后再单独对未遍历完的做操作
        while (l1 != null) {
            ListNode temp = new ListNode();
            if (carry) {
                temp.val = l1.val + 1;
            } else {
                temp.val = l1.val;
            }
            if (temp.val >= 10) {
                temp.val = temp.val - 10;
                carry = true;
            } else {
                carry = false;
            }
            l1 = l1.next;
            cur.next = temp;
            cur = cur.next;
        }
        while (l2 != null) {
            ListNode temp = new ListNode();
            if (carry) {
                temp.val = l2.val + 1;
            } else {
                temp.val = l2.val;
            }
            if (temp.val >= 10) {
                temp.val = temp.val - 10;
                carry = true;
            } else {
                carry = false;
            }
            l2 = l2.next;
            cur.next = temp;
            cur = cur.next;
        }
        if (carry) {
            ListNode temp = new ListNode(1);
            cur.next = temp;
            cur = cur.next;
        }
        return ansDummy.next;
    }
}
```

比较简单的题，直接看代码吧









## [0234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

```java
public boolean isPalindrome(ListNode head) {
    if (head.next == null) {
        return true;
    }
    Deque<Integer> deque = new LinkedList<>();
    ListNode fast = head, slow = head;
    deque.addLast(head.val);
    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        deque.addLast(slow.val);
    }
    //处理链表为单数情况
    if (fast.next == null) {
        deque.pollLast();
    }
    while (slow.next != null) {
        slow = slow.next;
        if (slow.val != deque.pollLast()) {
            return false;
        }
    }
    return true;
}
```

1. 利用快慢双指针来将slow指针定位到链表的中间位置，slow每次移动一格，fast每次移动两格，每次都把slow的元素放入栈中
2. 如果出循环时`fast.next == null`，说明链表元素数量是单数，此时的slow在正中间元素位置，回文格式是不需要考虑中间元素的，所以把栈顶元素出栈；如果`fast.next.next == null`，说明链表元素数量是双数，不需要对栈有额外操作。
3. 继续从slow开始遍历，每次移动一格，然后对比栈顶元素，看是否相同，不相同说明不是回文格式。



如果想要`O(n)` 时间复杂度和 `O(1)` 空间复杂度解决







## [0287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

```java
public int findDuplicate(int[] nums) {
    int slow = nums[0];
    int fast = nums[0];
    slow = nums[slow];
    fast = nums[nums[fast]];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[nums[fast]];
    }
    int loopStart = nums[0];
    while (loopStart != slow) {
        loopStart = nums[loopStart];
        slow = nums[slow];
    }
    return loopStart;
}
```

题目要求：

给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

解法：

图论，利用链表来实现本题解法，首先可以把数组首先模拟成链表。由于只存在一个，且一定有一个数字的数量大于1，因此链表中一定存在一个环，那么该题就变成了寻找链表环的入口。利用快慢指针即可解得答案。









# Two pointers

## [0015. 三数之和](https://leetcode-cn.com/problems/3sum/)

```java
//0015.三数之和
public class Solution32 {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        // 枚举 a
        for(int i=0; i<nums.length; i++){
            // 需要和上一次枚举的数不相同
            if(i>0&&nums[i]==nums[i-1]){
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int right=nums.length-1;
            // 枚举 b
            for(int left=i+1; left<nums.length; left++){
                // 需要和上一次枚举的数不相同
                if(left>i+1&&nums[left]==nums[left-1]){
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while(left<right&&nums[i]+nums[left]+nums[right]>0){
                    right--;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if(left==right){
                    break;
                }
                if(nums[i]+nums[left]+nums[right]==0){
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }
}

```

思路：

X数之和，时间复杂度就是O(n^(X - 1))，最后的两个数的遍历可以利用双指针法，但是之前的所有遍历只能是一层层循环遍历。

首先本题需要先对数组做一个排序操作，这个操作非常关键，因为排序操作可以为之后去重复操作打下基础，不排序的情况没法去重复。对于去重，所有元素除了最后一个元素，都要进行去重：

```java
			if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
```

1. 对于第一个数first，我们选取正常遍历`for (i = 0; i < nums.length; i++)`，每次循环进行一个去重判断，并将第三个元素（在本题中就是右指针）的指针位置定位到数组的末尾。

2. 对于第二个数second（在本题因为只有三个数之和，所以就是左指针），开始对左指针遍历，左指针每次都会被定位到第一个数右边一个位置，在这层循环中也是一样先做去重判断，然后就是左右两个指针的动态运动了：

   ```java
   				// 需要保证 b 的指针在 c 的指针的左侧
   				while (left < right && nums[i] + nums[left] + nums[right] > 0){
                       right--;
                   }
                   // 如果指针重合，随着 b 后续的增加
                   // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                   if(left == right){
                       break;
                   }
   ```

   这个左右指针的移动规则就是：

   1. 如果此时sum和大于0，那么说明sum需要变小才能满足总和为0，所以需要把右指针左移，因为数组已经被排序过了，左移右指针对应的元素值就是在变小。
   2. 如果此时左指针index等于右指针了或者sum和小于0了那么就退出循环。
   3. 之后循环走完，左指针向右移。







# DFS & BFS

## [0200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```java
//DFS
    public int numIslands1(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    checkNumIslands(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    private void checkNumIslands(char[][] grid, int i_index, int j_index) {
        if (i_index < 0 || i_index >= grid.length) {
            return;
        }
        if (j_index < 0 || j_index >= grid[0].length) {
            return;
        }
        if (grid[i_index][j_index] == '0') {
            return;
        }
        grid[i_index][j_index] = '0';
        checkNumIslands(grid, i_index - 1, j_index);
        checkNumIslands(grid, i_index + 1, j_index);
        checkNumIslands(grid, i_index, j_index - 1);
        checkNumIslands(grid, i_index, j_index + 1);
    }
```

### 解法一深度优先遍历DFS：

网格类问题的 DFS 遍历方法
网格问题的基本概念
我们首先明确一下岛屿问题中的网格结构是如何定义的，以方便我们后面的讨论。

网格问题是由 m \times nm×n 个小方格组成一个网格，每个小方格与其上下左右四个方格认为是相邻的，要在这样的网格上进行某种搜索。

岛屿问题是一类典型的网格问题。每个格子中的数字可能是 0 或者 1。我们把数字为 0 的格子看成海洋格子，数字为 1 的格子看成陆地格子，这样相邻的陆地格子就连接成一个岛屿。

![岛屿问题示例](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/c36f9ee4aa60007f02ff4298bc355fd6160aa2b0d628c3607c9281ce864b75a2.png)

在这样一个设定下，就出现了各种岛屿问题的变种，包括岛屿的数量、面积、周长等。不过这些问题，基本都可以用 DFS 遍历来解决。

DFS 的基本结构
网格结构要比二叉树结构稍微复杂一些，它其实是一种简化版的图结构。要写好网格上的 DFS 遍历，我们首先要理解二叉树上的 DFS 遍历方法，再类比写出网格结构上的 DFS 遍历。我们写的二叉树 DFS 遍历一般是这样的：

```java
void traverse(TreeNode root) {
    // 判断 base case
    if (root == null) {
        return;
    }
    // 访问两个相邻结点：左子结点、右子结点
    traverse(root.left);
    traverse(root.right);
}
```

可以看到，二叉树的 DFS 有两个要素：「访问相邻结点」和「判断 base case」。

第一个要素是访问相邻结点。二叉树的相邻结点非常简单，只有左子结点和右子结点两个。二叉树本身就是一个递归定义的结构：一棵二叉树，它的左子树和右子树也是一棵二叉树。那么我们的 DFS 遍历只需要递归调用左子树和右子树即可。

第二个要素是 判断 base case。一般来说，二叉树遍历的 base case 是 root == null。这样一个条件判断其实有两个含义：一方面，这表示 root 指向的子树为空，不需要再往下遍历了。另一方面，在 root == null 的时候及时返回，可以让后面的 root.left 和 root.right 操作不会出现空指针异常。

对于网格上的 DFS，我们完全可以参考二叉树的 DFS，写出网格 DFS 的两个要素：

首先，网格结构中的格子有多少相邻结点？答案是上下左右四个。对于格子 (r, c) 来说（r 和 c 分别代表行坐标和列坐标），四个相邻的格子分别是 (r-1, c)、(r+1, c)、(r, c-1)、(r, c+1)。换句话说，网格结构是「四叉」的。

![网格结构中四个相邻的格子](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/63f5803e9452ccecf92fa64f54c887ed0e4e4c3434b9fb246bf2b410e4424555.png)

其次，网格 DFS 中的 base case 是什么？从二叉树的 base case 对应过来，应该是网格中不需要继续遍历、grid[r][c] 会出现数组下标越界异常的格子，也就是那些超出网格范围的格子。

![网格 DFS 的 base case](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/5a91ec351bcbe8e631e7e3e44e062794d6e53af95f6a5c778de369365b9d994e.png)

这一点稍微有些反直觉，坐标竟然可以临时超出网格的范围？这种方法我称为「先污染后治理」—— 甭管当前是在哪个格子，先往四个方向走一步再说，如果发现走出了网格范围再赶紧返回。这跟二叉树的遍历方法是一样的，先递归调用，发现 root == null 再返回。

网格结构的 DFS 与二叉树的 DFS 最大的不同之处在于，遍历中可能遇到遍历过的结点。这是因为，网格结构本质上是一个「图」，我们可以把每个格子看成图中的结点，每个结点有向上下左右的四条边。在图中遍历时，自然可能遇到重复遍历结点。

这时候，DFS 可能会不停地「兜圈子」，永远停不下来，如下图所示：

![DFS 遍历可能会兜圈子（动图）](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/7fec64afe8ab72c5df17d6a41a9cc9ba3879f58beec54a8791cbf108b9fd0685.png)

所以遍历过的岛屿，可以把值赋值为‘0’或者新增也可以赋值为'2'表示为已经遍历过的岛屿，这样就不会重复遍历了。



### 解法二广度优先遍历BFS：

```java
//BFS
    public int numIslands2(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1'){
                    count++;
                    int[] coordinate = new int[2];
                    coordinate[0] = i;
                    coordinate[1] = j;
                    bfs(coordinate, grid);
                }
            }
        }
        return count;
    }
    private void bfs(int[] coordinate, char[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(coordinate);
        while (!queue.isEmpty()) {
            int[] newCoordinate = queue.poll();
            if (newCoordinate[0] >= 0 && newCoordinate[0] < grid.length && newCoordinate[1] >= 0 && newCoordinate[1] < grid[0].length && grid[newCoordinate[0]][newCoordinate[1]] == '1') {
                grid[newCoordinate[0]][newCoordinate[1]] = '2';
                queue.offer(new int[]{newCoordinate[0] - 1, newCoordinate[1]});
                queue.offer(new int[]{newCoordinate[0] + 1, newCoordinate[1]});
                queue.offer(new int[]{newCoordinate[0], newCoordinate[1] - 1});
                queue.offer(new int[]{newCoordinate[0], newCoordinate[1] + 1});
            }
        }
    }
```

主循环和思路一类似，不同点是在于搜索某岛屿边界的方法不同。
bfs 方法：

借用一个队列 queue，判断队列首部节点 (i, j) 是否未越界且为 1：

可以想象一下，DFS的==递归==就是走的深度，而BFS这种利用队列的方法，就是广度，因为每次都是先把某一个节点周边一圈的节点先遍历完成，才会到下一层去。

若是则置零（删除岛屿节点），并将此节点上下左右节点 `(i+1,j),(i-1,j),(i,j+1),(i,j-1)` 加入队列；
若不是则跳过此节点；
循环 pop 队列首节点，直到整个队列为空，此时已经遍历完此岛屿。





**结论：**

DFS与BFS相比，DFS用时更短。

















# Dijkstra's Algorithm

> Two ways to calculate:
>
> 1. E * logE (E means the number of edges)
> 2. N ^ 2 (N means the number of Nodes)
>
> You can check the example question 743. Network Delay Time for understanding the above two methods.

## [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)

**E * logE**

```java
public int networkDelayTime(int[][] times, int n, int k) {
    Map<Integer, List<Pair<Integer, Integer>>> next = new HashMap<>();
    for (int[] time : times) {
        int source = time[0];
        int target = time[1];
        int weight = time[2];
        List<Pair<Integer, Integer>> targetList = next.getOrDefault(source, new ArrayList<>());
        targetList.add(new Pair<>(target, weight));
        next.put(source, targetList);
    }
    PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>((o1, o2) -> o1.getValue() - o2.getValue());
    Set<Integer> visited = new HashSet<>();
    pq.offer(new Pair<>(k, 0));
    int ret = 0;
    while (!pq.isEmpty()) {
        Pair<Integer, Integer> poll = pq.poll();
        Integer pollNode = poll.getKey();
        Integer distance = poll.getValue();
        if (visited.contains(pollNode)) continue;
        visited.add(pollNode);
        ret = Math.max(ret, distance);
        List<Pair<Integer, Integer>> list = next.getOrDefault(pollNode, new ArrayList<>());
        for (Pair<Integer, Integer> node : list) {
            pq.offer(new Pair<>(node.getKey(), node.getValue() + distance));
        }
    }
    return visited.size() == n ? ret : -1;
}
```







## [778. Swim in Rising Water](https://leetcode.cn/problems/swim-in-rising-water/)









# Binary Search

## [0240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

```java
public boolean searchMatrix(int[][] matrix, int target) {
    int row = matrix.length, col = matrix[0].length;
    return findT(matrix, target, 0, row, 0, col);
}

private Boolean findT(int[][] matrix, int target, int rowL, int rowR, int colL, int colR) {
    if (rowL >= rowR || colL >= colR) {
        return false;
    }
    int midRow = (rowL + rowR) / 2;
    int midCol = (colL + colR) / 2;
    if (matrix[midRow][midCol] > target) {
        if (findT(matrix, target, rowL, midRow, colL, colR)) {
            return true;
        }
        if (findT(matrix, target, rowL, rowR, colL, midCol)) {
            return true;
        }
    } else if (matrix[midRow][midCol] < target) {
        if (findT(matrix, target, midRow + 1, rowR, colL, colR)) {
            return true;
        }
        if (findT(matrix, target, rowL, rowR, midCol + 1, colR)) {
            return true;
        }
    } else {
        return true;
    }
    return false;
}
```

以上写的思路是二分查找递归，因为是一个二维矩阵，此时的二分查找有两个方向，一个是x轴，一个是y轴，因此我的递归代码中有两个不同的走法。

**另一种二分查找**思路是-正常遍历y轴，x轴的查找使用二分查找：

```java
public boolean searchMatrix(int[][] matrix, int target) {
        for (int[] row : matrix) {
            int index = search(row, target);
            if (index >= 0) {
                return true;
            }
        }
        return false;
    }
public int search(int[] nums, int target) {
    int low = 0, high = nums.length - 1;
    while (low <= high) {
        int mid = (high - low) / 2 + low;
        int num = nums[mid];
        if (num == target) {
            return mid;
        } else if (num > target) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return -1;
}
```

**方法三**：Z 字形查找

从左下查找到右上，不免发现从左下出发的话，往上走是小于当前元素的数，而往右走是大于当前元素的数，因此能非常简单的遍历到目标元素：

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
        }
        return false;
    }
}
```





## [0774. 最小化去加油站的最大距离](https://leetcode.cn/problems/minimize-max-distance-to-gas-station/)

```java
public double minmaxGasDist2(int[] stations, int k) {
    double left = 0, right = 0;
    for (int i = 1; i < stations.length; i++) {
        right = Math.max(right, (double) (stations[i] - stations[i - 1]));
    }
    while (right - left > 1e-6) {
        double mid = (right + left) / 2;
        int count = 0;
        for (int i = 1; i < stations.length; i++) {
            double t = (double) (stations[i] - stations[i - 1]) / mid;
            count += Math.ceil(t) - 1;
        }
        if (count > k) left = mid;
        else right = mid;
    }
    return left;
}
```

二分法

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220921144438101.png" alt="image-20220921144438101" style="zoom:50%;" />

left和right分别一开始代表0，和当前数组中相邻元素的最大距离，我们只要在[left,right]中去找可以满足新添加的加油站数量小于等于k，并且在这个条件内最小的相邻距离即可：

```java
		for (int i = 1; i < stations.length; i++) {
            double t = (double) (stations[i] - stations[i - 1]) / mid;
            count += Math.ceil(t) - 1;
        }
```

mid的含义就是在添加一定数量加油站后，形成一个相邻距离均为mid的加油站连续体。

此时如果count > k说明不满足条件，只能去加大加油站之间的间距来减少需新增加油站数量

```java
if (count > k) left = mid;
```

如果count <= k说明满足了条件，我们可以再去尝试寻找更下的加油站间距

```java
else right = mid;
```







# Deque

## [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    Deque<Integer> deque = new LinkedList<>();
    int[] ans = new int[nums.length + 1 - k];
    for (int i = 0; i < nums.length; i++) {
        if (!deque.isEmpty() && deque.peekFirst() <= i - k) {
            deque.pollFirst();
        }
        while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
            deque.pollLast();
        }
        deque.offerLast(i);
        if (i >= k - 1) {
            ans[i + 1 - k] = nums[deque.peekFirst()];
        }
    }
    return ans;
}
```

双端单调队列，左边用于维护滑动窗口的最大值不超过k，因为deque中存储的是数组元素下标，所以只要index <= i - k即可；右边用于维护最大元素，小于新入元素的全部元素都出栈





## [1425. 带限制的子序列和](https://leetcode.cn/problems/constrained-subsequence-sum/)

```java
//O(N)
// 1. maintain a monotone decreasing queue by poping out back elements smaller than nums[i]
// 2. Check if head element is out of date (in terms of sliding window)
// 3. The current head element of the deque is maximum of the sliding window
public int constrainedSubsetSum3(int[] nums, int k) {
    //dp[i]: The maximum sum of a non-empty subsequence of that array ending with nums[i]
    int[] dp = new int[nums.length];
    Deque<Integer> deque = new LinkedList<>();//store index
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
        while (!deque.isEmpty() && deque.peekFirst() < i - k) {
            deque.pollFirst();
        }
        dp[i] = nums[i];
        if (!deque.isEmpty()) dp[i] = Math.max(dp[i], dp[deque.peekFirst()] + nums[i]);
        while (!deque.isEmpty() && dp[deque.peekLast()] <= dp[i]) {
            deque.pollLast();
        }
        max = Math.max(max, dp[i]);
        deque.offerLast(i);
    }
    return max;
}
```

https://www.youtube.com/watch?v=FSbFPH7ejHk

双端队列，左边界用来维护有效范围内的元素，超过k的dp[i]全部出队，右边界用来维护最大元素，小于当前元素的所有dp[i]全部出栈。





## [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

```java
public int longestSubarray(int[] nums, int limit) {
    int left = 0, right = 0;
    int ret = 0;
    TreeMap<Integer, Integer> treeMap = new TreeMap<Integer, Integer>((a,b) -> {return a.compareTo(b);});
    treeMap.put(nums[0], 1);
    while (right < nums.length) {
        while (treeMap.lastKey() - treeMap.firstKey() <= limit) {
            ret = Math.max(ret, right - left + 1);
            right++;
            if (right < nums.length) treeMap.put(nums[right], treeMap.getOrDefault(nums[right], 0) + 1);
            else break;
        }
        while (treeMap.lastKey() - treeMap.firstKey() > limit) {
            if (treeMap.get(nums[left]) == 1) {
                treeMap.remove(nums[left]);
            } else {
                treeMap.put(nums[left], treeMap.get(nums[left]) - 1);
            }
            left++;
        }
    }
    return ret;
}
```

第一种，使用TreeMap保持滑动窗口中的元素始终有序，这样treeMap.firstKey()永远是最小的元素，而treeMap.lastKey()永远是最大的元素。

首先一直移动右指针，把遍历到的元素塞入treeMap，直到`treeMap.lastKey() - treeMap.firstKey() > limit`时停止遍历。

当出现`treeMap.lastKey() - treeMap.firstKey() > limit`时，只要移动左指针，一直删除treeMap中的元素，直到重新满足题目条件即可。

```java
public int longestSubarray2(int[] nums, int limit) {
    Deque<Integer> dequeMax = new LinkedList<>();
    Deque<Integer> dequeMin = new LinkedList<>();
    dequeMax.offerLast(0);
    dequeMin.offerLast(0);
    int right = 0;
    int ret = 0;
    int max = nums[0], min = nums[0];
    for (int left = 0; left < nums.length; left++) {
        while (max - min <= limit) {
            ret = Math.max(ret, right - left + 1);
            right++;
            if (right == nums.length) break;
            while (!dequeMax.isEmpty() && nums[dequeMax.peekLast()] <= nums[right]) {
                dequeMax.pollLast();
            }
            dequeMax.offerLast(right);
            while (!dequeMin.isEmpty() && nums[dequeMin.peekLast()] >= nums[right]) {
                dequeMin.pollLast();
            }
            dequeMin.offerLast(right);
            max = nums[dequeMax.peekFirst()];
            min = nums[dequeMin.peekFirst()];
        }
        if (right == nums.length) {
            break;
        }
        if (!dequeMax.isEmpty() && dequeMax.peekFirst() == left) {
            dequeMax.pollFirst();
            max = nums[dequeMax.peekFirst()];
        }
        if (!dequeMin.isEmpty() && dequeMin.peekFirst() == left) {
            dequeMin.pollFirst();
            min = nums[dequeMin.peekFirst()];
        }
    }
    return ret;
}
```

第二种方法依旧是使用Deque，但这次使用两个Deque双向队列，一个保证单调递增维护最小值，一个保证单调递减维护最大值，当最大值与最小值的差值大于limit时，开始移动左指针，因为Deque中存储的都是index，所以比较左指针的值和Deque队列中头元素的值即可确定是否出队。





## [1499. 满足不等式的最大值](https://leetcode.cn/problems/max-value-of-equation/)

```java
public int findMaxValueOfEquation(int[][] points, int k) {
    Deque<Integer> deque = new LinkedList<>();
    int ret = Integer.MIN_VALUE;
    for (int j = 0; j < points.length; j++) {
        while (!deque.isEmpty() && -points[deque.peekFirst()][0] + points[j][0] > k) {
            deque.pollFirst();
        }
        if (!deque.isEmpty()) {
            ret = Math.max(ret, points[deque.peekFirst()][1] - points[deque.peekFirst()][0] + points[j][0] + points[j][1]);
        }
        while (!deque.isEmpty() && -points[j][0] + points[j][1] >= -points[deque.peekLast()][0] + points[deque.peekLast()][1]) {
            deque.pollLast();
        }
        deque.offerLast(j);
    }
    return ret;
}
```

循环里面的顺序：

1. 我们以j为主体来思考这道题，deque就是用来维护j之前的所有元素的顺序
2. deque是单调递减的，这样就可以找到范围内最大的x+y。同时左边界用来维护保证deque中元素不过期。



## [1562. 查找大小为 M 的最新分组](https://leetcode.cn/problems/find-latest-group-of-size-m/)

```java
public int findLatestStep(int[] arr, int m) {
    if (m == arr.length) {
        return m;
    }
    Deque<Integer> deque = new LinkedList<>();
    int[] arr2 = new int[arr.length + 1];
    int[] day = new int[arr2.length];
    System.arraycopy(arr, 0, arr2, 1, arr.length);
    for (int i = 1; i < day.length; i++) {
        day[arr2[i]] = i;
    }
    int ret = -1;
    for (int i = 1; i < arr2.length; i++) {
        while(!deque.isEmpty() && day[deque.peekLast()] <= day[i]) {
            deque.pollLast();
        }
        deque.addLast(i);
        if (i < m) continue;
        while (!deque.isEmpty() && deque.peekFirst() <= i - m) {
            deque.pollFirst();
        }
        int t = day[deque.peekFirst()];
        int leftDay = i - m > 0 ? day[i - m] : Integer.MAX_VALUE; // [1,n]
        int rightDay = i + 1 < arr2.length ? day[i + 1] : Integer.MAX_VALUE;
        if (leftDay > t && rightDay >t) {
            ret = Math.max(ret, Math.min(leftDay, rightDay) - 1);
        }
    }
    return ret;
}
```

![image-20220909192843501](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220909192843501.png)

该题我们大体思路是一个滑动窗口，我们首先要求得滑动窗口中的t(即滑动窗口中满足全部元素为1的最小天数)

我们要新建立一个arr2数组，它的构成是在arr数组的基础上，对其头部加入一个空元素，方便之后对应。arr2数组现在的意思就是在第i天去把arr2[i]位置变为1，把它改造成day数组，意义为第i个位置的元素在第day[i]天被改为1。

然后就是利用最大滑动窗口去维护满足m个连续1的最小天数：

```java
while(!deque.isEmpty() && day[deque.peekLast()] <= day[i]) {
            deque.pollLast();
        }
```







## [1696. 跳跃游戏 VI](https://leetcode.cn/problems/jump-game-vi/)

[xxxx] x

本题是一个动态规划，**dp[i]意为跳到i时经过的最大数字之和**，`dp[i] = max(dp[i - 1], dp[i - 2], ..., dp[i - k]) + nums[i]` 

本题寻找区间内最大dp[i]的方法可以用最大滑动窗口来维护

```java
public int maxResult(int[] nums, int k) {
    Deque<Integer> deque = new LinkedList<>();
    deque.offerLast(0);
    int[] dp = new int[nums.length];
    dp[0] = nums[0];
    for (int i = 1; i < nums.length; i++) {
        while (!deque.isEmpty() && deque.peekFirst() < i - k) {
            deque.pollFirst();
        }
        int max = dp[deque.peekFirst()];
        dp[i] = max + nums[i];
        while (!deque.isEmpty() && dp[deque.peekLast()] <= dp[i]) {
            deque.pollLast();
        }
        deque.offerLast(i);
    }
    return dp[dp.length - 1];
}
```





## [2398. 预算内的最多机器人数目](https://leetcode.cn/problems/maximum-number-of-robots-within-budget/)

[xxxx] x单调递减deque维护最大chargeTimes







## [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

```java
long[] presum = new long[nums.length + 1];
presum[0] = 0;
for (int i = 1; i < presum.length; i++) {
    presum[i] = presum[i - 1] + (long) nums[i - 1];
}
int right = 0;
Deque<Integer> deque = new LinkedList<>();
int min = Integer.MAX_VALUE;
while (right < presum.length) {
    while (!deque.isEmpty() && presum[right] - presum[deque.peekFirst()] >= (long) k) {
        min = Math.min(min, right - deque.peekFirst());
        deque.pollFirst();
    }
    while (!deque.isEmpty() && presum[deque.peekLast()] >= presum[right]) {
        deque.pollLast();
    }
    deque.offerLast(right);
    right++;
}
return min == Integer.MAX_VALUE ? -1 : min;
```

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220912125923496.png" alt="image-20220912125923496" style="zoom:50%;" />

因为本题是要找子数组和大于k的最短数组，因此看上图，若`presum[i] & presum[j]，j > i & presum[j] <= presum[i]`，若`presum[cur] - presum[j] >= k`成立则`presum[cur] - presum[i] >= k`也一定成立，那么此时一定会选j而不是i，因为j比i大，`cur - j < cur - i`。一直维护一个单调递增的双端队列来解决该题。

- 若`!deque.isEmpty() && presum[deque.peekLast()] >= presum[right]`则将队顶元素出队，并记录此时的数组长度，记录最小数组长度即可







# PriorityQueue

## [1801. 积压订单中的订单总数](https://leetcode.cn/problems/number-of-orders-in-the-backlog/)

```java
public int getNumberOfBacklogOrders(int[][] orders) {
    PriorityQueue<int[]> queueSell = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    PriorityQueue<int[]> queueBuy = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o2[0] - o1[0];
        }
    });
    for (int[] order : orders) {
        if (order[2] == 0) queueBuy.offer(order);
        if (order[2] == 1) queueSell.offer(order);
        while (!queueBuy.isEmpty() && !queueSell.isEmpty()) {
            int[] buy = queueBuy.peek();
            int[] sell = queueSell.peek();
            if (buy[0] >= sell[0]) {
                if (buy[1] > sell[1]) {
                    queueSell.poll();
                    buy[1] -= sell[1];
                } else if (buy[1] < sell[1]) {
                    queueBuy.poll();
                    sell[1] -= buy[1];
                } else {
                    queueSell.poll();
                    queueBuy.poll();
                }
            } else break;
        }
    }
    int amount = 0;
    while (!queueBuy.isEmpty()) {
        int[] buy = queueBuy.poll();
        amount += buy[1];
        amount %= (int) (1e9 + 7);
    }
    while (!queueSell.isEmpty()) {
        int[] sell = queueSell.poll();
        amount += sell[1];
        amount %= (int) (1e9 + 7);
    }
    return amount;
}
```







## [1942. 最小未被占据椅子的编号](https://leetcode.cn/problems/the-number-of-the-smallest-unoccupied-chair/)

```java
public int smallestChair(int[][] times, int targetFriend) {
    PriorityQueue<Integer> chairQueue = new PriorityQueue<>(new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o1 - o2;
        }
    });
    for (int i = 0; i < 10000; i++) {
        chairQueue.offer(i);
    }
    PriorityQueue<int[]> occupyQueue = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    int[][] newTimes = new int[times.length][3];
    for (int i = 0; i < times.length; i++) {
        newTimes[i][0] = times[i][0];
        newTimes[i][1] = times[i][1];
        newTimes[i][2] = i;
    }
    Arrays.sort(newTimes, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    int targetChair = -1;
    for (int i = 0; i < newTimes.length; i++) {
        while (!occupyQueue.isEmpty() && occupyQueue.peek()[0] <= newTimes[i][0]) {
            chairQueue.offer(occupyQueue.poll()[1]);
        }
        if (newTimes[i][2] == targetFriend) {
            targetChair = chairQueue.peek();
            break;
        }
        occupyQueue.offer(new int[]{newTimes[i][1], chairQueue.poll()});
    }
    return targetChair;
}
```

利用两个优先队列维护空闲椅子按顺序排列，以及到场人员的离场时间按顺序排队。





## [1882. 使用服务器处理任务](https://leetcode.cn/problems/process-tasks-using-servers/)

```java
public int[] assignTasks(int[] servers, int[] tasks) {
    int[] ans = new int[tasks.length];
    int[][] newServers = new int[servers.length][2];
    //[0]:serverWight [1]:serverId
    PriorityQueue<int[]> serverQueue = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
                return o1[1] - o2[1];
            }
            return o1[0] - o2[0];
        }
    });
    for (int i = 0; i < servers.length; i++) {
        newServers[i][0] = servers[i];
        newServers[i][1] = i;
        serverQueue.offer(newServers[i]);
    }
    //[0]:endTime [1]:serverWight [2]:serverId
    PriorityQueue<int[]> processQueue = new PriorityQueue<>(new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    Queue<Integer> taskQueue = new LinkedList<>();
    int time = 0;
    int ansIndex = 0;
    while (ansIndex < tasks.length) {
        while (!processQueue.isEmpty() && processQueue.peek()[0] <= time) {
            int[] poll = processQueue.poll();
            serverQueue.offer(new int[]{poll[1], poll[2]});
        }
        if (time < tasks.length) taskQueue.offer(time);
        while (!taskQueue.isEmpty() && !serverQueue.isEmpty()) {
            Integer taskPoll = taskQueue.poll();
            int[] serverPoll = serverQueue.poll();
            processQueue.offer(new int[]{time + tasks[taskPoll], serverPoll[0], serverPoll[1]});
            ans[ansIndex] = serverPoll[1];
            ansIndex++;
        }if (time < tasks.length) {
            time++;
        } else {
            time = processQueue.peek()[0];
        }
        
    }
    return ans;
}
```

本题的坑在于遍历tasks与遍历完tasks之后的时间增长模式不一样，在tasks没遍历完之前只能以time++的模式去进行时间增长，因为每个任务都在i开始任务，所以相当于最早是每隔1的时间间隔。在遍历完成tasks之后就没必要严格按每秒的时间去递增了，直接从processQueue队列里把当前最早结束任务的结束时间拿出来即可。









## [2102. 序列顺序查询](https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/)

```java
public class SORTracker {
    private int getCount;

    private PriorityQueue<Pair<String, Integer>> sceneQueue;

    private PriorityQueue<Pair<String, Integer>> excludeQueue;

    public SORTracker() {
        this.getCount = 0;
        this.sceneQueue = new PriorityQueue<>(new Comparator<Pair<String, Integer>>() {
            @Override
            public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
                if (o1.getValue().equals(o2.getValue())) {
                    return orderByDic(o1.getKey(), o2.getKey());
                }
                return o2.getValue() - o1.getValue();
            }
        });
        this.excludeQueue = new PriorityQueue<>(new Comparator<Pair<String, Integer>>() {
            @Override
            public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
                if (o1.getValue().equals(o2.getValue())) {
                    return -orderByDic(o1.getKey(), o2.getKey());
                }
                return -(o2.getValue() - o1.getValue());
            }
        });
    }

    public void add(String name, int score) {
        excludeQueue.offer(new Pair<>(name, score));
        while (excludeQueue.size() > getCount) {
            sceneQueue.offer(excludeQueue.poll());
        }
    }

    public String get() {
        Pair<String, Integer> mostBeauty = sceneQueue.poll();
        excludeQueue.offer(mostBeauty);
        getCount++;
        return mostBeauty.getKey();
    }

    public int orderByDic(String s1, String s2) {
        int i = 0;
        while (i < s1.length() && i < s2.length()) {
            if (s1.charAt(i) == s2.charAt(i)) {
                i++;
                continue;
            }
            return s1.charAt(i) - s2.charAt(i);
        }
        return s1.length() > s2.length() ? 1 : -1;
    }
}
```

Dual priorityQueue

一个优先队列用来维护当前get位置开始往后已经存在的元素

另一个优先队列用来存已经被





## [2386. 找出数组的第 K 大和](https://leetcode.cn/problems/find-the-k-sum-of-an-array/)

```java
public long kSum(int[] nums, int k) {
    PriorityQueue<Pair<Long, Integer>> sumQueue = new PriorityQueue<>((a, b) -> {
        return (int) (a.getKey() - b.getKey());
    });
    long maxSum = 0;
    for (int num : nums) {
        maxSum += (long) num > 0 ? (long) num : 0;
    }
    if (k == 1) return maxSum;
    for (int i = 0; i < nums.length; i++) {
        nums[i] = Math.abs(nums[i]);
    }
    Arrays.sort(nums);
    sumQueue.offer(new Pair<>((long) Math.abs(nums[0]), 0));
    int count = 0;
    while (!sumQueue.isEmpty()) {
        Pair<Long, Integer> poll = sumQueue.poll();
        Integer index = poll.getValue();
        Long sum = poll.getKey();
        //由于最早已经塞入了一个nums[0]，所以开始count实际是1，同等于k - 1
        if (count == k - 2) return maxSum - sum;
        count++;
        if (index + 1 < nums.length) {
            sumQueue.offer(new Pair<>(sum - (long) nums[index] + (long) nums[index + 1], index + 1));
            sumQueue.offer(new Pair<>(sum + (long) nums[index + 1], index + 1));
        }
    }
    return -1;
}
```

1. 将题目转变为求一个正数数组的第k小序列和
2. 将题目给定的nums变为绝对值数组，并从小到大排序，这样之后的一个模拟递归的过程可以动态塞入当前最小值

![image-20220920150553839](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220920150553839.png)

![image-20220920150629846](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220920150629846.png)

![image-20220920151222716](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220920151222716.png)





## [0774. 最小化去加油站的最大距离](https://leetcode.cn/problems/minimize-max-distance-to-gas-station/)

```java
double upperBound = (double) (stations[stations.length - 1] - stations[0]) / (k + 1);
PriorityQueue<Pair<Double, Integer>> diffQueue = new PriorityQueue<>(new Comparator<Pair<Double, Integer>>() {
    @Override
    public int compare(Pair<Double, Integer> o1, Pair<Double, Integer> o2) {
        if (o1.getKey() - o2.getKey() > 1e-6) return -1;
        if (o2.getKey() - o1.getKey() > 1e-6) return 1;
        return 0;
    }
});
for (int i = 1; i < stations.length; i++) {
    double distance = (double) (stations[i] - stations[i - 1]);
    int t = Math.max((int) Math.floor(distance / upperBound), 1);
    diffQueue.offer(new Pair<>(distance / t, t));
    k -= t - 1;
}
for (int i = 1; i <= k; i++) {
    Pair<Double, Integer> poll = diffQueue.poll();
    Double key = poll.getKey();
    Integer value = poll.getValue();
    diffQueue.offer(new Pair<>(key * value / (value + 1), value + 1));
}
return diffQueue.peek().getKey();
```

定义一个优先队列，队列里面存的Pair中key是该块目前每一份的间距，value是该块目前已经被分成的份数

优先队列的队首永远是当前最大的间距，所以增加在队首的块的份数以此来减少其每份的间距。









# Topological Sort

## [0207. 课程表](https://leetcode.cn/problems/course-schedule/)

```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    int[] inDegree = new int[numCourses];
    Map<Integer, List<Integer>> map = new HashMap<>();
    for (int i = 0; i < prerequisites.length; i++) {
        inDegree[prerequisites[i][0]]++;
        Integer preNum = prerequisites[i][1];
        Integer curNum = prerequisites[i][0];
        List<Integer> classList = map.getOrDefault(preNum, new ArrayList<>());
        classList.add(curNum);
        map.put(preNum, classList);
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < inDegree.length; i++) {
        if (inDegree[i] == 0) {
            deque.addLast(i);
        }
    }
    while (!deque.isEmpty()) {
        Integer pre = deque.pollFirst();
        numCourses--;
        List<Integer> curList = map.containsKey(pre) ? map.get(pre) : new ArrayList<>();
        for (Integer cNum : curList) {
            if (--inDegree[cNum] == 0) {
                deque.addLast(cNum);
            }
        }
    }
    if (numCourses == 0) {
        return true;
    }
    return false;
}
```

方法1：入度表（BFS）

1. 统计课程安排图中每个节点的入度，生成入度表
2. 使用队列来把所有入度为0的节点入队
3. 在第一批入度为0的节点入队后，开始循环，当队列为空时退出循环
4. 循环每次都将队列中的头元素出队，将该元素所对应的后置课程的入度-1，如果此时某门课程的入度为0则入队。同时在出队时让numCources--。
5. 在循环结束时判断numCources是否为0，不为0则说明不成立





## [0210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)

```java
public int[] findOrder(int numCourses, int[][] prerequisites) {
    int[] inDgree = new int[numCourses];
    Map<Integer, List<Integer>> map = new HashMap<>();
    for (int[] prerequisite : prerequisites) {
        inDgree[prerequisite[0]]++;
        List<Integer> classList = map.getOrDefault(prerequisite[1], new ArrayList<>());
        classList.add(prerequisite[0]);
        map.put(prerequisite[1], classList);
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < inDgree.length; i++) {
        if (inDgree[i] == 0) {
            deque.offerLast(i);
        }
    }
    int[] res = new int[numCourses];
    int count = 0;
    while (!deque.isEmpty()) {
        Integer preClass = deque.pollFirst();
        res[count] = preClass;
        count++;
        List<Integer> classList = map.containsKey(preClass) ? map.get(preClass) : new ArrayList<>();
        for (Integer clazz :classList){
            if (--inDgree[clazz] == 0) deque.offerLast(clazz);
        }
    }
    if (numCourses == count) return res;
    return new int[0];
}
```

和课程表1一模一样，只不过求的东西不一样





## [1462. 课程表 IV](https://leetcode.cn/problems/course-schedule-iv/)

```java
public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
    Map<Integer, List<Integer>> map = new HashMap<>();
    int[] inDgree = new int[numCourses];
    List<Set<Integer>> preCourses = new ArrayList<>();
    for (int[] prerequisite : prerequisites) {
        List<Integer> list = map.getOrDefault(prerequisite[0], new ArrayList<>());
        list.add(prerequisite[1]);
        map.put(prerequisite[0], list);
        inDgree[prerequisite[1]]++;
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < inDgree.length; i++) {
        Set<Integer> set = new HashSet<>();
        set.add(i);
        preCourses.add(set);
        if (inDgree[i] == 0) deque.offerLast(i);
    }
    while (!deque.isEmpty()) {
        Integer pre = deque.pollFirst();
        Set<Integer> set = preCourses.get(pre);
        List<Integer> clazzList = map.containsKey(pre) ? map.get(pre) : new ArrayList<>();
        for (Integer clazz : clazzList) {
            preCourses.get(clazz).addAll(set);
            if (--inDgree[clazz] == 0) deque.offerLast(clazz);
        }
    }
    List<Boolean> res = new ArrayList<>();
    for (int[] query : queries) {
        res.add(preCourses.get(query[1]).contains(query[0]));
    }
    return res;
}
```

该题在课程表1和课程表2的基础上有一改动就是本题一定不会有环导致课程安排不合理

题目给了一个queries想让我们查找query[0]是不是query[1]的前置课

还是利用拓扑排序，先求得入度表，以及一个邻接表，同时还要有一个记录每门课的所有前置课程的表（利用Set）

```java
	while (!deque.isEmpty()) {
        Integer pre = deque.pollFirst();
        Set<Integer> set = preCourses.get(pre);
        List<Integer> clazzList = map.containsKey(pre) ? map.get(pre) : new ArrayList<>();
        for (Integer clazz : clazzList) {
            //pre是clazz的前置课，pre的所有前置课以及它自己都是clazz的前置课
            preCourses.get(clazz).addAll(set);
            if (--inDgree[clazz] == 0) deque.offerLast(clazz);
        }
    }
```





## [1136. 并行课程](https://leetcode.cn/problems/parallel-courses/)

```java
public int minimumSemesters(int n, int[][] relations) {
    int[] inDegree = new int[n + 1];
    Map<Integer, List<Integer>> postCourse = new HashMap<>();
    for (int[] relation : relations) {
        inDegree[relation[1]]++;
        List<Integer> list = postCourse.getOrDefault(relation[0], new ArrayList<>());
        list.add(relation[1]);
        postCourse.put(relation[0], list);
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 1; i < inDegree.length; i++) {
        if (inDegree[i] == 0) deque.offerLast(i);
    }
    int minSemester = 0;
    while (!deque.isEmpty()) {
        int size = deque.size();
        n -= size;
        minSemester++;
        for (int i = 0; i < size; i++) {
            Integer curCourse = deque.pollFirst();
            List<Integer> list = postCourse.getOrDefault(curCourse, new ArrayList<>());
            for (Integer course : list) {
                if (--inDegree[course] == 0) deque.offerLast(course);
            }
        }
    }
    if (n != 0) {
        return -1;
    }
    return minSemester;
}
```

本题和前三题的解题框架一模一样，就不细讲了









## [2050. 并行课程 III](https://leetcode.cn/problems/parallel-courses-iii/)

```java
public int minimumTime(int n, int[][] relations, int[] time) {
    int[] inDegree = new int[n + 1];
    Map<Integer, List<Integer>> postCourse = new HashMap<>();
    for (int[] relation : relations) {
        inDegree[relation[1]]++;
        List<Integer> list = postCourse.getOrDefault(relation[0], new ArrayList<>());
        list.add(relation[1]);
        postCourse.put(relation[0], list);
    }
    PriorityQueue<Pair<Integer, Integer>> queue = new PriorityQueue<>((a, b) -> {return a.getValue() - b.getValue();});
    for (int i = 1; i < inDegree.length; i++) {
        if (inDegree[i] == 0) queue.offer(new Pair<>(i, time[i - 1]));
    }
    Pair<Integer, Integer> poll = new Pair<>(0, 0);
    while (!queue.isEmpty()) {
        poll = queue.poll();
        List<Integer> list = postCourse.getOrDefault(poll.getKey(), new ArrayList<>());
        for (Integer course : list) {
            if (--inDegree[course] == 0) queue.offer(new Pair<>(course, poll.getValue() + time[course - 1]));
        }
    }
    return poll.getValue();
}
```

本题和前面几题的区别在于，该题有向，有权图。该权指的是修完每门课所需要的时间，当一个节点的入度大于1时，该门课程可以开始修的时间取决于他两个前置课靠后结束课程的结束时间。

使用优先队列就可以保证让新加入的入度为0的课程一定是在其结束时间靠后的先行课结束后才加入队列。





## [0310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220927115906086.png" alt="image-20220927115906086" style="zoom:50%;" />

为了找到最小高度树，我们需要找到一个节点离其他节点的距离最平均，这样就能得到最小高度，所以我们可以从外圈开始把最外层的节点一层层剥掉，这样剩下那个就是距离外层节点最平均的节点。

由于本题是无向图，所以我们处理的节点是入度为1的节点，任何一个节点的最小入度一定是1。

```java
public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    if (n == 1) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        return res;
    }
    Map<Integer, List<Integer>> map = new HashMap<>();
    int[] inDegree = new int[n];
    for (int[] edge : edges) {
        inDegree[edge[0]]++;
        inDegree[edge[1]]++;
        List<Integer> list1 = map.getOrDefault(edge[0], new ArrayList<>());
        List<Integer> list2 = map.getOrDefault(edge[1], new ArrayList<>());
        list1.add(edge[1]);
        list2.add(edge[0]);
        map.put(edge[0], list1);
        map.put(edge[1], list2);
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 1) {
            deque.offerLast(i);
        }
    }
    List<Integer> res = new ArrayList<>();
    while (!deque.isEmpty()) {
        int size = deque.size();
        res = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            Integer cur = deque.pollFirst();
            List<Integer> list = map.getOrDefault(cur, new ArrayList<>());
            res.add(cur);
            for (Integer post : list) {
                if (--inDegree[post] == 1) deque.offerLast(post);
            }
        }
    }
    return res;
}
```

当n等于1时需要需要单独判断一下，需要返回结果是[0]而不是[]。







## [0802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/)

```java
public List<Integer> eventualSafeNodes(int[][] graph) {
    Map<Integer, List<Integer>> map = new HashMap<>();
    int[] outDegree = new int[graph.length];
    for (int i = 0; i< graph.length; i++) {
        for (int out : graph[i]) {
            outDegree[i]++;
            List<Integer> list = map.getOrDefault(out, new ArrayList<>());
            list.add(i);
            map.put(out, list);
        }
    }
    Deque<Integer> deque = new LinkedList<>();
    List<Integer> ans = new ArrayList<>();
    for(int i = 0; i < outDegree.length; i++) {
        if (outDegree[i] == 0) {
            deque.offerLast(i);
            ans.add(i);
        }
    }
    while (!deque.isEmpty()) {
        Integer poll = deque.pollFirst();
        List<Integer> list = map.getOrDefault(poll, new ArrayList<>());
        for (Integer pre : list) {
            if (--outDegree[pre] == 0) {
                deque.offerLast(pre);
                ans.add(pre);
            }
        }
    }
    Collections.sort(ans);
    return ans;
}
```

该题想找到可以走到终端节点的节点，终端节点也就是没有出度的节点。所以本题利用邻接表和出度表就可以完成。









## [2115. 从给定原材料中找到所有可以做出的菜](https://leetcode.cn/problems/find-all-possible-recipes-from-given-supplies/)

```java
public List<String> findAllRecipes(String[] recipes, List<List<String>> ingredients, String[] supplies) {
    Map<String, Integer> inDegree = new HashMap<>();
    Set<String> supplySet = new HashSet<>(Arrays.asList(supplies));
    Map<String, List<String>> map = new HashMap<>();
    for (int i = 0; i < recipes.length; i++) {
        inDegree.put(recipes[i], 0);
        for (String s : ingredients.get(i)) {
            if (!supplySet.contains(s)) {
                inDegree.put(recipes[i], inDegree.get(recipes[i]) + 1);
                List<String> list = map.getOrDefault(s, new ArrayList<>());
                list.add(recipes[i]);
                map.put(s, list);
            }
        }
    }
    Deque<String> deque = new LinkedList<>();
    Iterator<Map.Entry<String, Integer>> iterator = inDegree.entrySet().iterator();
    while (iterator.hasNext()) {
        Map.Entry<String, Integer> entry = iterator.next();
        if (entry.getValue() == 0) {
            deque.offerLast(entry.getKey());
        }
    }
    List<String> ans = new ArrayList<>();
    while (!deque.isEmpty()) {
        String poll = deque.pollFirst();
        ans.add(poll);
        List<String> list = map.getOrDefault(poll, new ArrayList<>());
        for (String s : list) {
            Integer indegree = inDegree.get(s);
            inDegree.put(s, --indegree);
            if (indegree == 0) deque.offerLast(s);
        }
    }
    return ans;
}
```

本题有原材料这么一说，原材料无限供应，所以但凡recipes中需要用到原材料我们可以默认这个入度不增加，只有碰到recipes中的ingredient是recipes时，我们才增加入度。

- 把suppiles改造成Set，因为我们在遍历每一个recipes的ingredient时，我们需要快速访问supplies中是否包含这个材料：
  - 如果有这个材料，则不做任何操作，因为该材料是无限供应的
  - 如果没有这个材料，则需要入度++，并添加进邻接表中
- 之后的操作就是拓扑排序了







## [2204. Distance to a Cycle in Undirected Graph](https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/)

```java
public int[] distanceToCycle(int n, int[][] edges) {
    Map<Integer, List<Integer>> nearTable = new HashMap<>();
    int[] inDegree = new int[n];
    for (int[] edge : edges) {
        inDegree[edge[0]]++;
        inDegree[edge[1]]++;
        List<Integer> list1 = nearTable.getOrDefault(edge[0], new ArrayList<>());
        list1.add(edge[1]);
        List<Integer> list2 = nearTable.getOrDefault(edge[1], new ArrayList<>());
        list2.add(edge[0]);
        nearTable.put(edge[0], list1);
        nearTable.put(edge[1], list2);
    }
    Deque<Integer> deque = new LinkedList<>();
    int[] ans = new int[n];
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 1) {
            deque.offerLast(i);
        }
    }
    while (!deque.isEmpty()) {
        Integer pollFirst = deque.pollFirst();
        ans[pollFirst] = -1;
        List<Integer> list = nearTable.getOrDefault(pollFirst, new ArrayList<>());
        for (Integer node : list) {
            if (--inDegree[node] == 1) deque.offerLast(node);
        }
    }
    // 反向BFS回去，因为已经找到了所有的环节点，从环节点再往外扩散出去
    for (int i = 0; i < ans.length; i++) {
        if (ans[i] == 0) deque.offerLast(i);
    }
    while (!deque.isEmpty()) {
        Integer pollFirst = deque.pollFirst();
        List<Integer> list = nearTable.getOrDefault(pollFirst, new ArrayList<>());
        for (Integer node : list) {
            // 不用ans[node] != -1可以让之后的循环正常进行，已遍历过的节点一定是大于等于0的，所以不会重复遍历
            if (ans[node] != -1) continue;
            ans[node] = ans[pollFirst] + 1;
            deque.offerLast(node);
        }
    }
    return ans;
}
```

本题首先要找到无向图中的环，这个就是拓扑排序从外一层层往里面找，环中节点的入度一定大于等于2，所以当遍历停止时，剩下没遍历到的节点就是环节点，并在这个遍历过程中，把所有非环节点标记为-1。第一个遍历结束后，此时已经知道所有环节点了，从环节点往外遍历即可求得每个点到环节点的最短距离。





## [2392. 给定条件下构造矩阵](https://leetcode.cn/problems/build-a-matrix-with-conditions/)







## [1857. 有向图中最大颜色值](https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/)







## [1203. 项目管理](https://leetcode.cn/problems/sort-items-by-groups-respecting-dependencies/)

```java
public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
    //先改造group数组，给没有明确组的任务分配组
    //于是下面就是从groupId等于m开始依次往上继续接手任务
    //TODO 由于我们要改变数组，最保险的办法就是把入度图用Map来实现，用普通的Array的话可能会出现越界问题
    int groupIndex = m;
    Map<Integer, List<Integer>> groupItems = new HashMap<>();
    for (int i = 0; i < group.length; i++) {
        if (group[i] == -1) {
            group[i] = groupIndex;
            groupIndex++;
        }
        List<Integer> list = groupItems.getOrDefault(group[i], new ArrayList<>());
        list.add(i);
        groupItems.put(group[i], list);
    }
    Map<Integer, List<Integer>> adjacencyTable = new HashMap<>();
    int[] inDegree = new int[n];
    for (int i = 0; i < beforeItems.size(); i++) {
        for (Integer pre : beforeItems.get(i)) {
            //只考虑组间的邻接关系，不同组直接跳过
            if (group[i] != group[pre]) continue;
            inDegree[i]++;
            List<Integer> list = adjacencyTable.getOrDefault(pre, new ArrayList<>());
            list.add(i);
            adjacencyTable.put(pre, list);
        }
    }
    //因为我们此时已经有每个组所拥有的任务列表了，我们可以利用之前任务之间的入度表和邻接表进行拓扑排序然后得到每个组按顺序排好的任务列表
    Map<Integer, List<Integer>> groupItemsOrdered = new HashMap<>();
    Iterator<Map.Entry<Integer, List<Integer>>> iterator = groupItems.entrySet().iterator();
    while (iterator.hasNext()) {
        Map.Entry<Integer, List<Integer>> entry = iterator.next();
        int groupId = entry.getKey();
        List<Integer> orderedList = topologySort(entry.getValue(), adjacencyTable, inDegree);
        if (orderedList.size() != groupItems.get(groupId).size()) return new int[]{};
        groupItemsOrdered.put(groupId, orderedList);
    }
    //重新构建入度和邻接表，用于排序group
    inDegree = new int[n];
    adjacencyTable.clear();
    for (int i = 0; i < n; i++) {
        for (Integer pre : beforeItems.get(i)) {
            if (group[i] == group[pre]) continue;
            List<Integer> preList = adjacencyTable.getOrDefault(group[pre], new ArrayList<>());
            if (!preList.contains(group[i])) {
                preList.add(group[i]);
                adjacencyTable.put(group[pre], preList);
                inDegree[group[i]]++;
            }
        }
    }
    //这里要用Set，因为要保证每一个group只出现一次，然后在下面传入拓扑排序的function中转成list即可
    Set<Integer> groupIds = new HashSet<>();
    for (int i = 0; i < n; i++) {
        groupIds.add(group[i]);
    }
    List<Integer> groupOrdered = topologySort(new ArrayList<>(groupIds), adjacencyTable, inDegree);
    List<Integer> res = new ArrayList<>();
    for (Integer groupId : groupOrdered) {
        for (Integer node : groupItemsOrdered.get(groupId)) {
            res.add(node);
        }
    }
    Integer[] integers = res.toArray(new Integer[res.size()]);
    return Arrays.stream(integers).mapToInt(Integer::valueOf).toArray();
}

private List<Integer> topologySort(List<Integer> groupItems, Map<Integer, List<Integer>> adjacencyTable, int[] inDegree) {
    Deque<Integer> deque = new LinkedList<>();
    for (Integer groupItem : groupItems) {
        if (inDegree[groupItem] == 0) deque.offerLast(groupItem);
    }
    List<Integer> res = new ArrayList<>();
    while (!deque.isEmpty()) {
        int cur = deque.pollFirst();
        res.add(cur);
        List<Integer> list = adjacencyTable.getOrDefault(cur, new ArrayList<>());
        for (Integer next : list) {
            if (--inDegree[next] == 0) deque.offerLast(next);
        }
    }
    if (res.size() != groupItems.size()) {
        return new ArrayList<>();
    } else {
        return res;
    }
}
```







## [1591. 奇怪的打印机 II](https://leetcode.cn/problems/strange-printer-ii/)

```java
public boolean isPrintable(int[][] targetGrid) {
    int[] right = new int[61];
    int[] left = new int[61];
    int[] bottom = new int[61];
    int[] top = new int[61];
    Arrays.fill(right, -1);
    Arrays.fill(left, 60);
    Arrays.fill(bottom, -1);
    Arrays.fill(top, 60);
    for (int i = 0; i < targetGrid.length; i++) {
        for (int j = 0; j < targetGrid[0].length; j++) {
            int color = targetGrid[i][j];
            right[color] = Math.max(right[color], j);
            left[color] = Math.min(left[color], j);
            bottom[color] = Math.max(bottom[color], i);
            top[color] = Math.min(top[color], i);
        }
    }
    int[] inDegree = new int[61];
    Map<Integer, List<Integer>> next = new HashMap<>();
    for (int i = 0; i < targetGrid.length; i++) {
        for (int j = 0; j < targetGrid[0].length; j++) {
            for (int color = 1; color <= 60; color++) {
                if (i >= top[color] && i <= bottom[color] && j >= left[color] && j <= right[color]) {
                    if (color != targetGrid[i][j]) {
                        inDegree[targetGrid[i][j]]++;
                    }
                    List<Integer> list = next.getOrDefault(color, new ArrayList<>());
                    list.add(targetGrid[i][j]);
                    next.put(color, list);
                }
            }
        }
    }
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 1; i < inDegree.length; i++) {
        if (inDegree[i] == 0) {
            deque.offerLast(i);
        }
    }
    while (!deque.isEmpty()) {
        Integer poll = deque.pollFirst();
        List<Integer> list = next.getOrDefault(poll, new ArrayList<>());
        for (Integer color : list) {
            if (--inDegree[color] == 0) deque.offerLast(color);
        }
    }
    for (int count : inDegree) {
        if (count > 0) return false;
    }
    return true;
}
```

我们可以首先先求得矩阵中每一个颜色的上下左右边界（即某个颜色在矩阵中的最左元素，最右元素，最上元素，最下元素，全部记录下来）。根据所有颜色的边界，我们对应每一个坐标，去找该坐标在哪些颜色的边界范围内，这些坐标都有可能曾经被这些颜色占据过。如果在(m, n)节点，该节点可以被a颜色占据，但其targetGrid矩阵中存的最终颜色是b颜色，并不是a颜色，那么久说明a颜色一定在b颜色之前占据该节点。因此本题就可以转化为图论，根据刚刚的逻辑构造入度和邻接表。最后只要遍历入度表，如果有元素值大于0，说明有环，那么返回false。







## [1632. 矩阵转换后的秩](https://leetcode.cn/problems/rank-transform-of-a-matrix/)

```java
public class Solution13 {
    int[] father;

    public int[][] matrixRankTransform(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        father = new int[m * n];
        initFather(m, n);
        Map<Integer, List<Integer>> next = new HashMap<>();
        //对于一个二维矩阵，其坐标可以用i(当前行下标) * m(二维矩阵最大列数) + j(当前列下标)来组成一维unicode，所以我们直接用一个一维数组做入度表
        int[] inDegree = new int[m * n];
        //本题用拓扑排序最难处理的是相同val的元素，因此我们需要得到一个group数组，里面记录了相同列相同行组成的相同val的union
        //先对每行进行处理
        for (int i = 0; i < m; i++) {
            List<Pair<Integer, Integer>> orderedVal = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                orderedVal.add(new Pair<>(matrix[i][j], i * n + j));
            }
            Collections.sort(orderedVal, (a, b) -> {
                return a.getKey() - b.getKey();
            });
            for (int j = 1; j < n; j++) {
                if (orderedVal.get(j - 1).getKey() == orderedVal.get(j).getKey()) {
                    if (findFather(orderedVal.get(j - 1).getValue()) != findFather(orderedVal.get(j).getValue())) {
                        unionAsGroup(orderedVal.get(j - 1).getValue(), orderedVal.get(j).getValue());
                    }
                } else {
                    List<Integer> list = next.getOrDefault(orderedVal.get(j - 1).getValue(), new ArrayList<>());
                    list.add(orderedVal.get(j).getValue());
                    next.put(orderedVal.get(j - 1).getValue(), list);
                    inDegree[orderedVal.get(j).getValue()]++;
                }
            }
        }
        //对每列进行处理
        for (int j = 0; j < n; j++) {
            List<Pair<Integer, Integer>> orderedVal = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                orderedVal.add(new Pair<>(matrix[i][j], i * n + j));
            }
            Collections.sort(orderedVal, (a, b) -> {
                return a.getKey() - b.getKey();
            });
            for (int i = 1; i < m; i++) {
                if (orderedVal.get(i - 1).getKey() == orderedVal.get(i).getKey()) {
                    if (findFather(orderedVal.get(i - 1).getValue()) != findFather(orderedVal.get(i).getValue())) {
                        unionAsGroup(orderedVal.get(i - 1).getValue(), orderedVal.get(i).getValue());
                    }
                } else {
                    List<Integer> list = next.getOrDefault(orderedVal.get(i - 1).getValue(), new ArrayList<>());
                    list.add(orderedVal.get(i).getValue());
                    next.put(orderedVal.get(i - 1).getValue(), list);
                    inDegree[orderedVal.get(i).getValue()]++;
                }
            }
        }
        //key: father idx, value: all members.
        Map<Integer, List<Integer>> group = new HashMap<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                List<Integer> list = group.getOrDefault(findFather(i * n + j), new ArrayList<>());
                list.add(i * n + j);
                group.put(i * n + j, list);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (father[i * n + j] != i * n + j) {
                    inDegree[father[i * n + j]] += inDegree[i * n + j];
                }
            }
        }
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (father[i * n + j] == i * n + j && inDegree[i * n + j] == 0) {
                    deque.offerLast(i * n + j);
                }
            }
        }
        int rank = 1;
        int[][] res = new int[m][n];
        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                Integer pollFirst = deque.pollFirst();
                //先给group中所有元素赋值为当前的rank
                List<Integer> groups = group.getOrDefault(pollFirst, new ArrayList<>());
                for (Integer member : groups) {
                    res[member / n][member % n] = rank;
                    List<Integer> list = next.getOrDefault(member, new ArrayList<>());
                    for (Integer node : list) {
                        if (--inDegree[father[node]] == 0) deque.offerLast(father[node]);
                    }
                }
            }
            rank++;
        }
        return res;
    }

    private void initFather(int m, int n) {
        for (int i = 0; i < m * n; i++) {
            father[i] = i;
        }
    }

    private void unionAsGroup(Integer x, Integer y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    private int findFather(Integer x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }
}
```







# Stack

## [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

```java
public int largestRectangleArea(int[] heights) {
    int size = heights.length;
    int[] nextSmaller = new int[size];
    Arrays.fill(nextSmaller, size);
    int[] prevSmaller = new int[size];
    Arrays.fill(prevSmaller, -1);
    Stack<Integer> stack = new Stack<>();
    for (int i = 0; i < size; i++) {
        while (!stack.empty() && heights[i] <= heights[stack.peek()]) {
            nextSmaller[stack.pop()] = i;
        }
        if (!stack.empty()) {
            prevSmaller[i] = stack.peek();
        }
        stack.push(i);
    }
    int max = 0;
    for (int i = 0; i < size; i++) {
        int area = heights[i] * (nextSmaller[i] - prevSmaller[i] - 1);
        max = Math.max(max, area);
    }
    return max;
}
```







## [2334. 元素值大于变化阈值的子数组](https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/)

```java
public int validSubarraySize(int[] nums, int threshold) {
    int size =  nums.length;
    int[] nextSmaller = new int[size];
    Arrays.fill(nextSmaller, size);
    int[] prevSmaller = new int[size];
    Arrays.fill(prevSmaller, -1);
    Stack<Integer> stack = new Stack<>();
    for (int i = 0; i < size; i++) {
        while (!stack.empty() && nums[i] <= nums[stack.peek()]) {
            nextSmaller[stack.pop()] = i;
        }
        if (!stack.empty()) {
            prevSmaller[i] = stack.peek();
        }
        stack.push(i);
    }
    for (int i = 0; i < size; i++) {
        int k = nextSmaller[i] - prevSmaller[i] - 1;
        if (nums[i] > threshold / k) {
            return k;
        }
    }
    return -1;
}
```

利用stack来维护一个单调递增的序列，因为单调递增，当新元素进入时，先把stack中的元素出栈到队首元素小于新元素为止，这个过程中出栈的所有元素的nextSmaller就是新元素，而该新元素的prevSmaller就是更新完后的stack的队首元素（因为是单调增队列，更新完后的队首元素一定比新元素小）。





## [85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/)

不断地移动纵坐标，这样就会有m种不同的heights数组，这样就完全变回了84题

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20221015145218260.png" alt="image-20221015145218260" style="zoom:50%;" />

```java
int length = matrix[0].length;
int max = 0;
for (int i = matrix.length - 1; i >= 0; i--) {
    int[] nextSmaller = new int[length];
    int[] prevSmaller = new int[length];
    Arrays.fill(nextSmaller, length);
    Arrays.fill(prevSmaller, -1);
    Stack<Integer> stack = new Stack<>();
    int[] heights = new int[length];
    for (int j = 0; j < length; j++) {
        int height = 0;
        for (int k = i; k >= 0; k--) {
            if (matrix[k][j] == '0') {
                break;
            }
            height++;
        }
        heights[j] = height;
    }
    for (int j = 0; j < length; j++) {
        while (!stack.empty() && heights[stack.peek()] >=heights[j]) {
            nextSmaller[stack.pop()] = j;
        }
        if (!stack.empty()) {
            prevSmaller[j] = stack.peek();
        }
        stack.push(j);
    }
    for (int j = 0; j < length; j++) {
        int area = heights[j] * (nextSmaller[j] - prevSmaller[j] - 1);
        max = Math.max(max, area);
    }
}
return max;
```









## [1944. 队列中可以看到的人数](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/)

有 n 个人排成一个队列，从左到右 编号为 0 到 n - 1 。给你以一个整数数组 heights ，每个整数 互不相同，heights[i] 表示第 i 个人的高度。

一个人能 看到 他右边另一个人的条件是这两人之间的所有人都比他们两人 矮 。更正式的，第 i 个人能看到第 j 个人的条件是 i < j 且 min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1]) 。

请你返回一个长度为 n 的数组 answer ，其中 answer[i] 是第 i 个人在他右侧队列中能 看到 的 人数 。

**解题思路**

对于该题，i位置的人可以看到的右边的人一定是一个单调递增队列，所以我们这题就去维护一个单调递减队列：

1. 当入队元素大于栈顶元素，则栈顶元素出栈，因为该元素大于栈顶元素，所以它是出栈元素所在位置的人所能看到的最远的人，其能看到的人数++
2. 元素入队时，栈顶元素一定可以看到该位置的人，所以栈顶元素位置的人所能看到的人数++

```java
public int[] canSeePersonsCount(int[] heights) {
    int length = heights.length;
    Stack<Integer> stack = new Stack<>();
    int[] ans = new int[length];
    for (int i = 0; i < length; i++) {
        while (!stack.empty() && heights[stack.peek()] < heights[i]) {
            ans[stack.peek()]++;
        }
        if (!stack.empty()) {
            //Need to consider if stack is equal this situation
            if (heights[stack.peek()] == heights[i]) {
                ans[stack.pop()]++;
            } else {
                ans[stack.peek()]++;
            }
        }
        stack.push(i);
    }
    return ans;
}
```



I have a few packages that have been stored in your warehouse for a while, and I'd like to know if you have a delivery service.

My package should be large in size, could you send it directly to my door.









## [2282. Number of People That Can Be Seen in a Grid](https://leetcode.cn/problems/number-of-people-that-can-be-seen-in-a-grid/)

```java
public int[][] seePeople(int[][] heights) {
    int m = heights.length;
    int n = heights[0].length;
    int[][] ans = new int[m][n];
    for (int i = 0; i < m; i++) {
        Stack<Integer> rolStack = new Stack<>();
        for (int j = 0; j < n; j++) {
            while (!rolStack.empty() && heights[i][rolStack.peek()] < heights[i][j]) {
                ans[i][rolStack.pop()]++;
            }
            if (!rolStack.empty()) {
                //Need to consider if stack.peek is equal to current element
                if (heights[i][rolStack.peek()] == heights[i][j]){
                    ans[i][rolStack.pop()]++;
                } else {
                    ans[i][rolStack.peek()]++;
                }
            }
            rolStack.push(j);
        }
    }
    for (int j = 0; j < n; j++) {
        Stack<Integer> colStack = new Stack<>();
        for (int i = 0; i < m; i++) {
            while (!colStack.empty() && heights[colStack.peek()][j] <= heights[i][j]) {
                ans[colStack.pop()][j]++;
            }
            if (!colStack.empty()) {
                if (heights[colStack.peek()][j] == heights[i][j]) {
                    ans[colStack.pop()][j]++;
                } else {
                    ans[colStack.peek()][j]++;
                }
            }
            colStack.push(i);
        }
    }
    return ans;
}
```







## [2197. 替换数组中的非互质数](https://leetcode.cn/problems/replace-non-coprime-numbers-in-array/)

```java
public List<Integer> replaceNonCoprimes(int[] nums) {
    Deque<Integer> stack = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
        int temp = nums[i];
        while (!stack.isEmpty()) {
            Integer pre = stack.peekLast();
            int gcd = findGCD(pre, temp);
            if (gcd <= 1) break;
            temp = pre / gcd * temp;
            stack.pollLast();
        }
        stack.offerLast(temp);
    }
    List<Integer> ans = new ArrayList<>();
    while (!stack.isEmpty()) {
        ans.add(stack.pollFirst());
    }
    return ans;
}

private int findGCD(int pre, int cur) {
    int bigger = Math.max(pre, cur);
    int smaller = Math.min(pre, cur);
    while (bigger % smaller != 0) {
        int temp = bigger % smaller;
        bigger = smaller;
        smaller = temp;
    }
    return smaller;
}
```

逐渐理解stack的妙用，将元素一个个塞入stack中，每次塞入都去和stack中栈顶元素进行是否互质的判断，如果互质则求出最大公约数，并弹出栈顶元素，塞入其最小公倍数。









## [2355. Maximum Number of Books You Can Take](https://leetcode.cn/problems/maximum-number-of-books-you-can-take/)









# Union Find

## Template

```java
public class Template {
    int[] father;
    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }
}
```





## [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)

```java
public class Solution1 {
    int[] father;
    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public int minimumEffortPath(int[][] heights) {
        int ans1 = binarySearch(heights);
        int ans2 = unionFind(heights);
        return ans1;
    }

    private int binarySearch(int[][] heights) {
        int left = 0, right = 1000000;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (isOk(heights, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean isOk(int[][] heights, int threshold) {
        int m = heights.length;
        int n = heights[0].length;

        Queue<Pair<Integer, Integer>> queue = new LinkedList<>();
        queue.offer(new Pair<>(0, 0));
        int[][] visited = new int[m][n];
        visited[0][0] = 1;
        int[][] dirs = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        while (!queue.isEmpty()) {
            Pair<Integer, Integer> poll = queue.poll();
            int y = poll.getKey();
            int x = poll.getValue();
            for (int[] dir : dirs) {
                int newY = y + dir[0];
                int newX = x + dir[1];
                if (newY < 0 || newY >= m || newX < 0 || newX >= n) {
                    continue;
                }
                if (visited[newY][newX] == 1) {
                    continue;
                }
                if (Math.abs(heights[newY][newX] - heights[y][x]) > threshold) {
                    continue;
                }
                queue.offer(new Pair<>(newY, newX));
                visited[newY][newX] = 1;
            }
        }
        return visited[m - 1][n - 1] == 1;
    }

    private int unionFind(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        father = new int[m * n];
        for (int i = 0; i < m * n; i++) {
            father[i] = i;
        }
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != n - 1) {
                    edges.add(new int[]{Math.abs(heights[i][j] - heights[i][j + 1]), i * n + j, i * n + j + 1});
                }
                if (i != m - 1) {
                    edges.add(new int[]{Math.abs(heights[i][j] - heights[i + 1][j]), i * n + j, (i + 1) * n + j});
                }
            }
        }
        Collections.sort(edges, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        for (int[] edge : edges) {
            if (findFather(edge[1]) != findFather(edge[2])) {
                union(edge[1], edge[2]);
            }
            if (findFather(0) == findFather(m * n - 1)) {
                return edge[0];
            }
        }
        return 0;
    }
}
```

包含了两种解，一种是用二分法，一种是用并查集。

二分法：

二分法可以用来快速找到最小的体力消耗值，每次都用二分法找到一个值来判断该值是否可行，如果不可行说明值小了，如果可行说明值够用，可以再往小的找。判断可行的办法就是利用BFS去走整个图，如果能从左上角走到右下角说明可行，否则不可行。

并查集：





## [1970. 你能穿过矩阵的最后一天](https://leetcode.cn/problems/last-day-where-you-can-still-cross/)

```java
public class Solution2 {
    int[] father;
    int m, n;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public int latestDayToCross(int row, int col, int[][] cells) {
        m = row;
        n = col;
        //本题的祖宗要多两个，为了方便检索是否有从最顶端到最底端的路径，所以第一行的所有节点都会被预处理连入一个节点a，最后一行的所有节点也会被预处理连入一个节点b，最后只要看a,b是否union即可
        father = new int[m * n + 2];
        for (int i = 0; i < m * n + 2; i++) {
            father[i] = i;
        }
        for (int j = 0; j < n; j++) {
            union(0 * n + j, m * n);
            union((m - 1) * n + j, m * n + 1);
        }
        int[][] mat = new int[m][n];
        for (int[] cell : cells) {
            mat[cell[0] - 1][cell[1] - 1] = 1;
        }
        int[][] dir = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 1) {
                    continue;
                }
                for (int k = 0; k < 4; k++) {
                    int y = i + dir[k][0];
                    int x = j + dir[k][1];
                    //如果越界则不管
                    if (y < 0 || y >= m || x < 0 || x >= n) {
                        continue;
                    }
                    //如果也是海洋则不管
                    if (mat[y][x] == 1) {
                        continue;
                    }
                    if (findFather(i * n + j) != findFather(y * n + x)) {
                        union(i * n + j, y * n + x);
                    }
                }
            }
        }
        for (int t = cells.length - 1; t >= 0; t--) {
            if (checkOK()) return t + 1;
            int i = cells[t][0] - 1;
            int j = cells[t][1] - 1;
            mat[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                int y = i + dir[k][0];
                int x = j + dir[k][1];
                //如果越界则不管
                if (y < 0 || y >= m || x < 0 || x >= n) {
                    continue;
                }
                //如果也是海洋则不管
                if (mat[y][x] == 1) {
                    continue;
                }
                if (findFather(i * n + j) != findFather(y * n + x)) {
                    union(i * n + j, y * n + x);
                }
            }
        }
        return 0;
    }

    private boolean checkOK() {
        return findFather(m * n) == findFather(m * n + 1);
    }
}
```

本题同样有两种解法，Binary Search和Union Find

Union Find:

本题采用回溯重构法，首先把时间变为最后一天，建立一个二维矩阵，标记0为陆地，1为水域。遍历整个将所有的陆地union在一起，然后一步步回溯时间，每一天都有一片水域变成陆地，把新多出来的陆地进行union。每一次回溯前都去检查是否有能从第一行的陆地能连到最后一行的陆地。







## [803. 打砖块](https://leetcode.cn/problems/bricks-falling-when-hit/)

```java
public class Solution3 {
    int[] father;
    int[] size;
    int[][] map;
    int[][] dir;
    int m, n;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
            size[x] += size[y];
        } else {
            father[x] = y;
            size[y] += size[x];
        }
    }

    public int[] hitBricks(int[][] grid, int[][] hits) {
        m = grid.length;
        n = grid[0].length;
        map = grid;
        father = new int[m * n];
        size = new int[m * n];
        for (int i = 0; i < m * n; i++) {
            father[i] = i;
            size[i] = 1;
        }
        for (int[] hit : hits) {
            if (map[hit[0]][hit[1]] == 0) {
                map[hit[0]][hit[1]] = -1;
            } else {
                map[hit[0]][hit[1]] = 0;
            }
        }
        dir = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (map[i][j] != 1) {
                    continue;
                }
                for (int k = 0; k < 4; k++) {
                    int y = i + dir[k][0];
                    int x = j + dir[k][1];
                    if (y < 0 | y >= m | x < 0 | x >= n) {
                        continue;
                    }
                    if (map[y][x] != 1) {
                        continue;
                    }
                    if (findFather(i * n + j) != findFather(y * n + x)) {
                        union(i * n + j, y * n + x);
                    }
                }
            }
        }
        List<Integer> reversedAns = new ArrayList<>();
        for (int i = hits.length - 1; i >= 0; i--) {
            if (map[hits[i][0]][hits[i][1]] == -1) {
                reversedAns.add(0);
                continue;
            }
            boolean isStable = false;
            int count = 0;
            for (int k = 0; k < 4; k++) {
                int y = hits[i][0] + dir[k][0];
                int x = hits[i][1] + dir[k][1];
                if (y < 0 | y >= m | x < 0 | x >= n) {
                    continue;
                }
                if (map[y][x] != 1) {
                    continue;
                }
                if (findFather(hits[i][0] * n + hits[i][1]) != findFather(y * n + x)) {
                    //如果回溯的这个砖头原本就在第一行，那么也要把isStable变为true
                    if (findFather(y * n + x) < n || hits[i][0] == 0) {
                        isStable = true;
                    }
                    if (findFather(y * n + x) >= n) {
                        count += size[findFather(y * n + x)];
                    }
                    union(hits[i][0] * n + hits[i][1], y * n + x);
                }
            }
            map[hits[i][0]][hits[i][1]] = 1;
            reversedAns.add(isStable ? count : 0);
        }
        Collections.reverse(reversedAns);
        return reversedAns.stream().mapToInt(Integer::valueOf).toArray();
    }
}
```







## [1697. 检查边长度限制的路径是否存在](https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/)

```java
//1697. 检查边长度限制的路径是否存在
public class Solution4 {
    int[] father;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
        father = new int[n];
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        //首先根据distance对edgeList进行排序
        Arrays.sort(edgeList, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }
        });
        //重构queries，记录此时每个节点的index，在排序之后（index会被打乱，如果不记录则无法知道原来的index），方便构造ans数组
        int[][] newQueries = new int[queries.length][queries[0].length + 1];
        for (int i = 0; i < queries.length; i++) {
            for (int j = 0; j < 3; j++) {
                newQueries[i][j] = queries[i][j];
            }
            newQueries[i][3] = i;
        }
        //当limit变大，只会有更多的边被union，所以根据查询数组的limit进行排序，这样之前的状态可以全部保留
        Arrays.sort(newQueries, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }
        });
        int index = 0;
        boolean[] ans = new boolean[queries.length];
        for (int[] query : newQueries) {
            //此时queries已经根据limit从小到大排序好了，edgeList也根据distance从小到大排序好了，我们只要在遍历edgeList直到某个边distance大于等于limit时停止即可
            while (index < edgeList.length && edgeList[index][2] < query[2]) {
                int node1 = edgeList[index][0];
                int node2 = edgeList[index][1];
                if (findFather(node1) != findFather(node2)) {
                    union(node1, node2);
                }
                index++;
            }
            //每一轮的union完成后，看查询数组中两个节点是否union即可
            ans[query[3]] = (findFather(query[0]) == findFather(query[1]));
        }
        return ans;
    }
}
```









## [2421. 好路径的数目](https://leetcode.cn/problems/number-of-good-paths/)

![image-20221109143135666](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20221109143135666.png)

```java
//2421. 好路径的数目
public class Solution5 {
    int[] father;
    //由于本题是路径描述的是一棵树，所以我们默认从上往下，也就是节点小的在前，Map的key就是节点的val，存放所有以val大的节点打头，另一个端点是val小的节点所构成的路径
    Map<Integer, List<Pair<Integer, Integer>>> e;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public int numberOfGoodPaths(int[] vals, int[][] edges) {
        int length = vals.length;
        father = new int[length];
        for (int i = 0; i < length; i++) {
            father[i] = i;
        }
        e = new HashMap<>();
        //遍历所有的边，用e来记录
        for (int[] edge : edges) {
            int a = edge[0], b = edge[1];
            //始终记录val大的节点，边信息记录从大端到小端，一层层剥进去
            if (vals[a] < vals[b]) {
                int temp = a;
                a = b;
                b = temp;
            }
            List<Pair<Integer, Integer>> list = e.getOrDefault(vals[a], new ArrayList<>());
            list.add(new Pair<>(a, b));
            e.put(vals[a], list);
        }
        //val2idx把所有val相同的节点存到一起
        Map<Integer, List<Integer>> val2idx = new HashMap<>();
        //valSet记录所有出现过的val值
        Set<Integer> valSet = new TreeSet<>();
        for (int i = 0; i < length; i++) {
            valSet.add(vals[i]);
            List<Integer> list = val2idx.getOrDefault(vals[i], new ArrayList<>());
            list.add(i);
            val2idx.put(vals[i], list);
        }
        int ret = 0;
        for (Integer val : valSet) {
            List<Pair<Integer, Integer>> list = e.getOrDefault(val, new ArrayList<>());
            for (Pair<Integer, Integer> pair : list) {
                Integer a = pair.getKey();
                Integer b = pair.getValue();
                if (findFather(a) != findFather(b)) {
                    union(a, b);
                }
            }
            //countMap用来记录所有的连通域
            Map<Integer, Integer> countMap = new HashMap<>();
            List<Integer> val2idxList = val2idx.getOrDefault(val, new ArrayList<>());
            for (Integer node : val2idxList) {
                //root就是每个连通区域的编号
                int root = findFather(node);
                Integer count = countMap.getOrDefault(root, 0);
                count++;
                countMap.put(root, count);
            }
            Iterator<Map.Entry<Integer, Integer>> iterator = countMap.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<Integer, Integer> next = iterator.next();
                int freq = next.getValue();
                ret += freq * (freq - 1) / 2;
            }
        }
        return ret + length;
    }

    public static void main(String[] args) {
        int[] i1 = new int[]{1, 3, 2, 1, 3};
        int[][] i2 = new int[][]{{0, 1}, {0, 2}, {2, 3}, {2, 4}};
        Solution5 s = new Solution5();
        System.out.println(s.numberOfGoodPaths(i1, i2));
    }
}
```







## [1627. 带阈值的图连通性](https://leetcode.cn/problems/graph-connectivity-with-threshold/)

```java
//1627. 带阈值的图连通性
public class Solution7 {
    int[] father;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public List<Boolean> areConnected(int n, int threshold, int[][] queries) {
        father = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            father[i] = i;
        }
        //从threshold开始一个个遍历，把threshold的倍数全部和threshold链接起来，并且把这些倍数标记，避免之后的重复操作
        int[] visited = new int[n + 1];
        for (int i = threshold + 1; i <= n; i++) {
            if (visited[i] == 1) continue;
            for (int j = 2 * i; j <= n; j += i) {
                visited[j] = 1;
                if (findFather(j) != findFather(i)) {
                    union(i, j);
                }
            }
        }
        List<Boolean> ans = new ArrayList<>();
        for (int[] query : queries) {
            ans.add(findFather(query[0]) == findFather(query[1]));
        }
        return ans;
    }
}
```









## [952. 按公因数计算最大组件大小](https://leetcode.cn/problems/largest-component-size-by-common-factor/)

![image-20221111151207816](https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20221111151207816.png)

根据质因数来做分组

```java
//952. 按公因数计算最大组件大小
public class Solution6 {
    int[] father;

    int findFather(int x) {
        if (father[x] != x) {
            father[x] = findFather(father[x]);
        }
        return father[x];
    }

    void union(int x, int y) {
        x = father[x];
        y = father[y];
        if (x < y) {
            father[y] = x;
        } else {
            father[x] = y;
        }
    }

    public List<Integer> eratosthenes(int n) {
        int[] q = new int[n + 1];
        List<Integer> primes = new ArrayList<>();
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (q[i] == 1) {
                continue;
            }
            int j = i * 2;
            while (j <= n) {
                q[j] = 1;
                j += i;
            }
        }
        for (int i = 2; i <= n; i++) {
            if (q[i] == 0) {
                primes.add(i);
            }
        }
        return primes;
    }

    public int largestComponentSize(int[] nums) {
        father = new int[100001];
        for (int i = 0; i < 100001; i++) {
            father[i] = i;
        }
        //两个数的最小公因数一定是一个质数，所以我们只要把sqrt(max)这个范围内的所有质数找出来，然后依次遍历
        List<Integer> primes = eratosthenes((int) Math.sqrt(100000));
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            for (Integer prime : primes) {
                if (num < prime) {
                    break;
                }
                while (num % prime == 0) {
                    if (findFather(nums[i]) != findFather(prime)) {
                        union(nums[i], prime);
                    }
                    num /= prime;
                }
                //如果此时num还没有被除尽，说明num是一个大于sqrt(max)范围的质数，因此单独在做一次union
                if (num > 1) {
                    if (findFather(num) != findFather(nums[i])) {
                        union(num, nums[i]);
                    }
                }
            }
        }
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int num : nums) {
            Integer get = countMap.getOrDefault(findFather(num), 0);
            countMap.put(findFather(num), get + 1);
        }
        int ret = 0;
        Iterator<Map.Entry<Integer, Integer>> iterator = countMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> next = iterator.next();
            ret = Math.max(ret, next.getValue());
        }
        return ret;
    }
}
```







## [1998. 数组的最大公因数排序](https://leetcode.cn/problems/gcd-sort-of-an-array/)

本题可以参考952，思路基本上一模一样。

我们想办法把所有拥有大于1的公因数的组合union到一块，然后对数组进行一次排序并用另一个数据结构保留之前的index，检查排序完后的index和之前的index是否在相同的连通域，如果是则说明可以互换，反之亦然。

```java
int[] father;

int findFather(int x) {
    if (father[x] != x) {
        father[x] = findFather(father[x]);
    }
    return father[x];
}

void union(int x, int y) {
    x = father[x];
    y = father[y];
    if (x < y) {
        father[y] = x;
    } else {
        father[x] = y;
    }
}

public List<Integer> eratosthenes(int n) {
    int[] q = new int[n + 1];
    List<Integer> primes = new ArrayList<>();
    for (int i = 2; i <= Math.sqrt(n); i++) {
        if (q[i] == 1) {
            continue;
        }
        int j = i * 2;
        while (j <= n) {
            q[j] = 1;
            j += i;
        }
    }
    for (int i = 2; i <= n; i++) {
        if (q[i] == 0) {
            primes.add(i);
        }
    }
    return primes;
}

public boolean gcdSort(int[] nums) {
    father = new int[100001];
    for (int i = 0; i < father.length; i++) {
        father[i] = i;
    }
    List<Integer> newNums = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
        newNums.add(nums[i]);
    }
    List<Integer> primes = eratosthenes((int) Math.sqrt(1e5));
    for (Integer num : newNums) {
        Integer val = num;
        for (Integer prime : primes) {
            if (prime > val) break;
            while (val % prime == 0) {
                if (findFather(num) != findFather(prime)) {
                    union(num, prime);
                }
                val /= prime;
            }
        }
        if (val > 1) {
            if (findFather(num) != findFather(val)) {
                union(num, val);
            }
        }
    }
    Collections.sort(newNums, new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o1 - o2;
        }
    });
    for (int i = 0; i < newNums.size(); i++) {
        if (findFather(newNums.get(i)) != findFather(nums[i])) {
            return false;
        }
    }
    return true;

```







# Math

## [0069. x 的平方根 ](https://leetcode.cn/problems/sqrtx/)

refer to https://leetcode.cn/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/



## [0912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

```java
	int[] res;//用于排序
    public int[] sortArray(int[] nums) {
        res = new int[nums.length];
        sortNums(nums, 0, nums.length - 1);
        return nums;
    }

    private void sortNums(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = (start + end) / 2;
        sortNums(nums, start, mid);
        sortNums(nums, mid + 1, end);
        int i = start, j = mid + 1;
        int index = 0;
        while (i <= mid && j <= end) {
            if (nums[i] <= nums[j]) {
                res[index++] = nums[i++];
            } else {
                res[index++] = nums[j++];
            }
        }
        while (i <= mid) {
            res[index++] = nums[i++];
        }
        while (j <= end) {
            res[index++] = nums[j++];
        }
        for (int k = start; k <= end; k++) {
            nums[k] = res[k - start];
        }
    }
```

归并排序：

利用了分治的思想，对一个长度为n的待排序序列，把其分解为两个长度为n/2的子序列。每次都调用递归函数使两个子序列有序，然后再进行线性合并，使整个子序列有序。

定义`sortNums(nums, start, end)`表示对nums数组里[start, end]的部分进行排序，流程如下：

1. 递归函数`sortNums(nums, start, mid)`，对nums数组里[start, mid]部分进行排序
2. 递归函数`sortNums(nums, mid + 1, end)`，对nums数组里[mid + 1, end]部分进行排序
3. 此时两部分子数组已经被排序完成[start, mid], [mid + 1, end]，我们可以对两个有序区间线性归并即可使[start, end]有序







## [0048. 旋转图像](https://leetcode.cn/problems/rotate-image/)

```java
	public void rotate(int[][] matrix) {
        int start = 0, end = matrix.length - 1;
        for (int i = 0; start < end; i++) {
            start = i;
            end = matrix.length - 1 - i;
            for (int j = 0; start + j < end; j++) {
                int temp = matrix[start][start + j];
                matrix[start][start + j] = matrix[end - j][start];
                matrix[end - j][start] = matrix[end][end - j];
                matrix[end][end - j] = matrix[start + j][end];
                matrix[start + j][end] = temp;
            }
        }
    }
```

这种改变结构就是自外向内顺时针循环，最后一个变的一定要提前先用temp保存下来，不然等循环到的时候已经被其他数据覆盖掉了。







## [0155. 最小栈](https://leetcode.cn/problems/min-stack/)

```java
public class MinStack {
    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    private ListNode head;

    private ListNode tail;

    private int size = 0;

    private int min;

    public MinStack() {
        head = new ListNode();
        tail = head;
        min = Integer.MAX_VALUE;
    }

    public void push(int val) {
        ListNode newNode = new ListNode(val);
        min = Math.min(min, val);
        if (size == 0) {
            head = newNode;
            tail = head;
        } else {
            tail.next = newNode;
            tail = tail.next;
        }
        size++;
    }
    public void pop() {
        if (size <= 0) {
            return;
        }
        min = Integer.MAX_VALUE;
        ListNode temp = head;
        while (temp.next != null) {
            tail = temp;
            min = Math.min(temp.val, min);
            temp = temp.next;
        }
        tail.next = null;
        size--;
    }

    public int top() {
        return tail.val;
    }

    public int getMin() {
        if (size <= 0) {
            return -1;
        }
        return min;
    }

    public static void main(String[] args) {
        MinStack m = new MinStack();
        m.push(-3);
        m.push(0);
        m.push(-2);
        System.out.println(m.getMin());
        m.pop();
        System.out.println(m.top());
        System.out.println(m.getMin());
    }
}
```

这题首先是对栈的一个实现

- 利用链表实现一个动态的栈
- 因为需要找到当前栈的最小值，所以需要有一个最小值min的属性
- 当入栈时，对比入栈元素和min的大小
- 当出栈时，因为是用链表实现的动态栈，本来就需要通过head头节点去找到倒数第二个节点，需要遍历一遍链表，所以在遍历的过程中重新进行一次最小值的寻找。





## [204. 计数质数](https://leetcode.cn/problems/count-primes/)

Eratosthenes    时间复杂度N * loglogN

2: 4, 6, 8, 10.....

3: 6, 9, 12, 15.....

5: 10, 15, 20, 25....

质数的倍数一定不是质数

```java
//204. 计数质数
public class Primes {
    public List<Integer> eratosthenes(int n) {
        //从2开始遍历，一旦找到某个数是质数，直接把其所有小于sqrt(n)的倍数都提前找出来，这样可以提前筛选掉很多质数，防止重复工作
        int[] q = new int[n + 1];
        List<Integer> primes = new ArrayList<>();
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (q[i] == 1) {
                continue;
            }
            int j = i * 2;
            while (j <= n) {
                q[j] = 1;
                j += i;
            }
        }
        for (int i = 2; i <= n; i++) {
            if (q[i] == 0) {
                primes.add(i);
            }
        }
        return primes;
    }
}
```









## [1819. 序列中不同最大公约数的数目](https://leetcode.cn/problems/number-of-different-subsequences-gcds/)

```
for a given x:
	find all multipliers of x in nums
	check if their gcd is x

//先求出每个元素的因数
nums[1] = a => f1, f2, f3....
nums[2] = b => f2, f3, f4....
	//存放因数的倍数的集合
	multiplier[f1] = {a}
	multiplier[f2] = {a, b}
	multiplier[f3] = {a, b}
	multiplier[f4] = {b}
```

```java
//1819. 序列中不同最大公约数的数目
public class Solution1 {
    public int countDifferentSubsequenceGCDs(int[] nums) {
        //g[x]: gcd of all multiplier of x
        int[] g = new int[200001];
        for (int num : nums) {
            for (int i = 1; i <= Math.sqrt(num); i++) {
                if (num % i == 0) {
                    //if g[i] == 0, num is the first multiplier of i, so now the gcd is num itself
                    if (g[i] == 0) {
                        g[i] = num;
                    } else {
                        g[i] = getGCD(g[i], num);
                    }
                    //因为我们循环只到sqrt(num)，因此i是num的因数，num/i也是num因数
                    int another = num / i;
                    if (g[another] == 0) {
                        g[another] = num;
                    } else {
                        g[another] = getGCD(g[another], num);
                    }
                }
            }
        }
        int count = 0;
        //g[]中所有的元素都已经记录了nums中其对应的倍数，并且已经动态记录了gcd，所以只要遍历一次g[]，如果i的所有倍数的最大公约数就是i，那么i就是这个子序列的最大公约数
        for (int i = 1; i <= 200000; i++) {
            if (g[i] == i) {
                count++;
            }
        }
        return count;
    }

    //欧几里得算法
    private int getGCD(int m, int n) {
        int r = m % n;
        while (r != 0) {
            m = n;
            n = r;
            r = m % n;
        }
        return n;
    }
}
```









## [2183. 统计可以被 K 整除的下标对数目](https://leetcode.cn/problems/count-array-pairs-divisible-by-k/)

```
题目要求:
nums[i] * nums[j] % k == 0
我们可以先求nums[i]和k的最大公约数 -> a = gcd(nums[i], k)
b = k / a
我们需要找到b为nums[j]的约数的所有nums集合

举个很简单的例子，nums[i]已经是k的倍数了，那么此时a = gcd(nums[i], k) = k
因此b = k / a = k / k = 1
那么此时nums中任意元素都可以和nums[i]组合满足nums[i] * nums[j] % k == 0

或者nums[i]和k没有公约数，那么a = gcd(nums[i], k) = 1
因此b = k / a = k /1 = k
接下来就要找nums[j]能够被k整除

解法:
1.求得k所有的约数，只要满足nums[i]和nums[j]的约数相乘能得到k即可，也就是k的约束中选出两个相乘为k的结果
2.把所有约数为factor的元素下标存入map
```

```java
//2183. 统计可以被 K 整除的下标对数目
public class Solution2 {
    private int getGCD(int m, int n) {
        int r = m % n;
        while (r != 0) {
            m = n;
            n = r;
            r = m % n;
        }
        return n;
    }

    public long countPairs(int[] nums, int k) {
        Set<Integer> factors = new HashSet<>();
        for (int i = 1; i <= Math.sqrt(k); i++) {
            if (k % i == 0) {
                factors.add(i);
                factors.add(k / i);
            }
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            for (Integer factor : factors) {
                //把所有约数为factor的元素下标存入map
                if (nums[i] % factor == 0) {
                    List<Integer> list = map.getOrDefault(factor, new ArrayList<>());
                    list.add(i);
                    map.put(factor, list);
                }
            }
        }
        long count = 0;
        for (int i = 0; i < nums.length; i++) {
            int a = getGCD(nums[i], k);
            int b = k / a;
            //此时需要找到下标大于i的元素位置，利用二分法可以减少时间复杂度，从该位置往后的所有元素都满足题目要求，求出其子序列长度即可
            if (!map.containsKey(b)) continue;
            List<Integer> list = map.get(b);
            int left = 0, right = list.size() - 1;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (list.get(mid) <= i) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            if (list.get(left) <= i) {
                left++;
            }
            count += list.size() - left;
        }
        return count;
    }
}
```









## [2344. 使数组可以被整除的最少删除次数](https://leetcode.cn/problems/minimum-deletions-to-make-array-divisible/)

```java
//2344. 使数组可以被整除的最少删除次数
public class Solution3 {
    private int getGCD(int m, int n) {
        int r = m % n;
        while (r != 0) {
            m = n;
            n = r;
            r = m % n;
        }
        return n;
    }

    public int minOperations(int[] nums, int[] numsDivide) {
        Arrays.sort(nums);
        //先求出numsDivide整个数组的最大公约数
        int gcd = numsDivide[0];
        for (int num : numsDivide) {
            gcd = getGCD(gcd, num);
        }
        int count = 0;
        //num只要是gcd的因数就可以整除整个numsDivide数组
        for (int num : nums) {
            if (gcd % num == 0) {
                return count;
            }
            count++;
        }
        return -1;
    }
}
```







## [296. 最佳的碰头地点](https://leetcode.cn/problems/best-meeting-point/)

```
min |x1 - x| + |y1 - y| + |x2 - x| + |y2 - y| + ... + |xn - x| + |yn - y|
x = argmin |x1 - x| + |x2 - x| + ... + |xn - x|
  = median of {xi}
y = argmin |y1 - y| + |y2 - y| + ... + |yn - y|
  = median of {yi}
```

```java
//296. 最佳的碰头地点
public class Solution4 {
    public int minTotalDistance(int[][] grid) {
        List<Integer> x = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    x.add(j);
                    y.add(i);
                }
            }
        }
        Collections.sort(x);
        Collections.sort(y);
        int xMedian = x.get(x.size() / 2);
        int yMedian = y.get(y.size() / 2);
        int sumX = 0;
        for (Integer xi : x) {
            sumX += Math.abs(xi - xMedian);
        }
        int sumY = 0;
        for (Integer yi : y) {
            sumY += Math.abs(yi - yMedian);
        }
        return sumX + sumY;
    }
}
```









## [462. 最小操作次数使数组元素相等 II](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/)

```java
//462. 最小操作次数使数组元素相等 II
public class Solution5 {
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int median = nums[nums.length / 2];
        int count = 0;
        for (int num : nums) {
            count += Math.abs(num - median);
        }
        return count;
    }
}
```

做最少次+1或者-1操作，让所有元素相等，其实就是希望让总变化次数最小，那么就求中位数即可。







## [2448. 使数组相等的最小开销](https://leetcode.cn/problems/minimum-cost-to-make-array-equal/)

```java
相较于462，本题每个元素都被赋予了权值，每进行一次调整的花费都不同
x = argmin a * |x1 - x| + b * |x2 - x| + ... + z * |xn - x|
直接看公式，我们就没法取x1 - xn的中位数当作其最终值使总开销最小
但我们可以对公式进行变形
a * |x1 - x| = |x1 - x| + |x1 - x| + ... + |x1 - x|把它理解为有a个x1
这样就变成求a个x1，b个x2...z个xn的中位数了
```

```java
public long minCost(int[] nums, int[] cost) {
    //改造nums，因为我们要对其进行排序，所以把原本的数组下标保存进去
    int[][] newNums = new int[nums.length][2];
    for (int i = 0; i < nums.length; i++) {
        newNums[i][0] = nums[i];
        newNums[i][1] = cost[i];
    }
    Arrays.sort(newNums, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    long amount = 0;
    for (int[] newNum : newNums) {
        amount += newNum[1];
    }
    long medianIndex = amount / 2;
    long median = 0;
    long temp = 0;
    for (int[] newNum : newNums) {
        temp += newNum[1];
        //这里注意，逻辑上是index，而temp此时并不是index形式，因此应该-1变为index表达形式
        if (temp - 1 >= medianIndex) {
            median = newNum[0];
            break;
        }
    }
    long ret = 0;
    for (int[] newNum : newNums) {
        ret += Math.abs(newNum[0] - median) * (long) newNum[1];
    }
    return ret;
}
```











## [2033. 获取单值网格的最小操作数](https://leetcode.cn/problems/minimum-operations-to-make-a-uni-value-grid/)

```
|1 3 5| |7 9 10| |12 13 14| -> 3 9 13
|1 9 11|  |2 7 8|  |3 4 5| -> 1 2 3 4 5 7 8 9 11 -> 9 7 4
```

```java
//2033. 获取单值网格的最小操作数
public class Solution7 {
    public int minOperations(int[][] grid, int x) {
        int remainder = grid[0][0] % x;
        List<Integer> array = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] % x != remainder) {
                    return -1;
                }
                array.add(grid[i][j]);
            }
        }
        Collections.sort(array);
        int ret = 0;
        int median = array.get(array.size() / 2);
        for (Integer num : array) {
            ret += Math.abs(num - median) / x;
        }
        return ret;
    }
}
```







## [1703. Minimum Adjacent Swaps for K Consecutive Ones](https://leetcode.cn/problems/minimum-adjacent-swaps-for-k-consecutive-ones/)

由于本题不再是把所有节点并为一个点，而是把所有节点排成一条直线，所以我们需要给结果一个补偿。

```java
//计算补偿
private static int getCompensate(int k) {
    int compensate = 0;
    int a = k - 1;
    if (a % 2 != 0) {
        compensate += (a + 1) / 2;
        a--;
    }
    int b = a / 2;
    if (b > 1) {
        compensate += ((1 + b) * b) / 2 * 2;
    } else {
        compensate += b * 2;
    }
    return compensate;
}
```

在滑动窗口中，如果每次都重新遍历整个滑动窗口，那么会超时。因此我们可以动态更新最新的最小移动值，方法如下

```
0  1  2  mid  .  .  k-1  k
X  X  X   0   X  X   X        =>sum1
   X  X   X   0  X   X   X    =>sum2
   
1. - abs(p[mid]-p[0])首先减去头元素的距离差
2. + abs(p[k]-p[mid+1])加上尾元素的距离差
3. + k/2 * (p[mid+1]-p[mid])前k/2个元素此时产生了一段新的距离
4. - (k - 1 - k/2) * (p[mid+1]-p[mid])后(k-1-k/2)个元素此时少了一段距离，这里防止奇偶问题，不能用k/2-1
```

代码：

```java
//1703. Minimum Adjacent Swaps for K Consecutive Ones
public class Solution8 {
    public int minMoves(int[] nums, int k) {
        List<Integer> only1 = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                only1.add(i);
            }
        }
        int left = 0, right = 0;
        //我们不再是把所有节点并为一个点，而是把所有节点排成一条直线，所以我们需要给结果一个补偿。
        int compensate = getCompensate(k);
        int sum = 0;
        for (int i = 0; i < k; i++) {
            sum += Math.abs(only1.get(i) - only1.get(k / 2));
        }
        int ret = sum;
        for (int i = 0; i + k < only1.size(); i++) {
            int mid = i + k / 2;
            sum -= Math.abs(only1.get(mid) - only1.get(i));
            sum += Math.abs(only1.get(i + k) - only1.get(mid + 1));
            sum += (k / 2) * Math.abs(only1.get(mid) - only1.get(mid + 1));
            sum -= (k - 1 - k / 2) * Math.abs(only1.get(mid + 1) - only1.get(mid));
            ret = Math.min(ret, sum);
        }
        return ret - compensate;
    }

    private static int getCompensate(int k) {
        int compensate = 0;
        int a = k - 1;
        if (a % 2 != 0) {
            compensate += (a + 1) / 2;
            a--;
        }
        int b = a / 2;
        if (b > 1) {
            compensate += ((1 + b) * b) / 2 * 2;
        } else {
            compensate += b * 2;
        }
        return compensate;
    }
}
```









# Divide & Conque

## Example

```
A:[Y Y Y Y Y Y Z Z Z Z Z]
B:[Y Y Y Y Y Y] C:[Z Z Z Z Z]
Example: counts[i] is the number of smaller elements to the right of A[i]
分治法(归并排序):
我们可以把A数组拆为B和C两个数组，对B数组计算countsB[i] is the number of smaller elements to the right of B[i]，C数组做相同操作。
结果来说C数组的结果就是A数组中对应位置元素的结果，但B数组的结果并不是A数组最后的结果，需要在C数组的基础上进行更新。
**为了减少时间复杂度，我们可以在分治后，在返回到上级前，对B和C进行排序，这样在返回到上级后，我们可以用二分查找找到C数组中有哪些元素比B[i]小**
```

## [315. Count of Smaller Numbers After Self](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/)

```java
//315. Count of Smaller Numbers After Self
public class Solution1 {
    List<Integer> ret = new ArrayList<>();

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            ret.add(0);
        }
        int[] sorted = Arrays.copyOf(nums, n);
        divideConque(sorted, nums, 0, n - 1);
        return ret;
    }

    private void divideConque(int[] sorted, int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        divideConque(sorted, nums, start, mid);
        divideConque(sorted, nums, mid + 1, end);
        for (int i = start; i <= mid; i++) {
            //用二分法进行查找
            int left = mid + 1;
            int right = end;
            while (left < right) {
                int bsMid = left + (right - left) / 2;
                if (sorted[bsMid] >= nums[i]) {
                    right = bsMid;
                } else {
                    left = bsMid + 1;
                }
            }
            //如果在二分查找开始前，left已经等于right了(此时已经整理好的数组长度仅为1)，那么就需要手动判断nums[i]是否大于这个数，如果大于，让count++
            if (nums[i] > sorted[left]) left++;
            Integer count = ret.get(i);
            count += left - (mid + 1);
            ret.set(i, count);
        }
        Arrays.sort(sorted, start, end + 1);
    }
}
```









## [327. Count of Range Sum](https://leetcode.cn/problems/count-of-range-sum/) & 二分查找边界问题可以看这题

关于lower_bound和upper_bound的实现，本题可以理解为，前缀和之差就是两个点之间的range sum，因此对前缀和进行归并排序即可

```java
//327. Count of Range Sum
public class Solution2 {
    int l, u;

    int ret = 0;

    long[] temp;

    public int countRangeSum(int[] nums, int lower, int upper) {
        l = lower;
        u = upper;
        temp = new long[100001];
        int n = nums.length;
        //前缀和需要多给一个元素，第一个元素的前缀和是0
        long[] presums = new long[n + 1];
        for (int i = 0; i < n; i++) {
            presums[i + 1] = presums[i] + nums[i];
        }
        long[] sorted = Arrays.copyOf(presums, n + 1);
        divideConque(sorted, presums, 0, n);
        return ret;
    }

    private void divideConque(long[] sorted, long[] presums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        divideConque(sorted, presums, start, mid);
        divideConque(sorted, presums, mid + 1, end);
        mergeSorted(sorted, presums, start, end, mid);
    }

    private void mergeSorted(long[] sorted, long[] presums, int start, int end, int mid) {
        for (int i = start; i <= mid; i++) {
            int right = binarySearchRight(sorted, presums[i], mid + 1, end);
            int left = binarySearchLeft(sorted, presums[i], mid + 1, end);
            ret += right - left + 1 > 0 ? right - left + 1 : 0;
        }
        //归并排序，temp是一个临时数组，用来记录左右子数组排序后的状态
        int index1 = start, index2 = mid + 1, p = 0;
        while (index1 <= mid && index2 <= end) {
            if (sorted[index1] <= sorted[index2]) {
                temp[p] = sorted[index1];
                index1++;
            } else {
                temp[p] = sorted[index2];
                index2++;
            }
            p++;
        }
        while (index1 <= mid) {
            temp[p] = sorted[index1];
            index1++;
            p++;
        }
        while (index2 <= end) {
            temp[p] = sorted[index2];
            index2++;
            p++;
        }
        for (int i = 0; i < end - start + 1; i++) {
            sorted[i + start] = temp[i];
        }
    }

    //二分查找寻找左边界，开区间请改成sorted[mid] > target + l即可
    private int binarySearchLeft(long[] sorted, long target, int start, int end) {
        int left = start, right = end;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sorted[mid] >= target + l) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (right < start) {
            return start;
        }
        if (left > end) {
            return end + 1;
        }
        return left;
    }

    //二分查找寻找右边界，开区间请改成sorted[mid] < target + u即可
    private int binarySearchRight(long[] sorted, long target, int start, int end) {
        int left = start, right = end;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sorted[mid] <= target + u) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < start) {
            return start - 1;
        }
        if (left > end) {
            return end;
        }
        return right;
    }
}
```











## [493. Reverse Pairs](https://leetcode.cn/problems/reverse-pairs/)

```java
//493. Reverse Pairs
public class Solution3 {
    int ret = 0;

    int[] temp;

    public int reversePairs(int[] nums) {
        int n = nums.length;
        int[] sorted = Arrays.copyOf(nums, n);
        temp = new int[50001];
        divideConque(sorted, nums, 0, n - 1);
        return ret;
    }

    private void divideConque(int[] sorted, int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        divideConque(sorted, nums, start, mid);
        divideConque(sorted, nums, mid + 1, end);
        mergeSorted(sorted, nums, start, end, mid);
    }

    private void mergeSorted(int[] sorted, int[] nums, int start, int end, int mid) {
        for (int i = start; i <= mid; i++) {
            int index = binarySearch(sorted, nums[i], mid + 1, end);
            ret += index - (mid + 1) + 1;
        }
        int index1 = start, index2 = mid + 1, p = 0;
        while (index1 <= mid && index2 <= end) {
            if (sorted[index1] >= sorted[index2]) {
                temp[p] = sorted[index2];
                index2++;
            } else {
                temp[p] = sorted[index1];
                index1++;
            }
            p++;
        }
        while (index1 <= mid) {
            temp[p] = sorted[index1];
            index1++;
            p++;
        }
        while (index2 <= end) {
            temp[p] = sorted[index2];
            index2++;
            p++;
        }
        for (int i = 0; i < end - start + 1; i++) {
            sorted[i + start] = temp[i];
        }
    }

    private int binarySearch(int[] sorted, int num, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            //找右边界，同时是开区间，所以不用<=。同时注意类型转换，本题2 * Integer_MAXVALUE会出错的，应该转成long类型
            if (2 * (long) sorted[mid] < (long) num) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right;
    }
}
```







## [1649. Create Sorted Array through Instructions](https://leetcode.cn/problems/create-sorted-array-through-instructions/)

```
归并排序
对C:|YYYYYYZZZZZZ|分治
A:|YYYYYY|  B:|ZZZZZZZ|
解决A问题后，A在C中也已经是正确解了，但是解决B问题后，在C问题中要考虑A对B的影响，因此
		for (int i = mid + 1; i <= end; i++) {
            // "5" |1 2 3 * 5 5 * 7 8 9|
            //严格小于instructions[i]，所以是求左边界闭区间
            int left = binarySearchL(sorted, instructions[i], start, mid);
            smaller[i] += left - start;
        }
遍历B，对B中每一个元素x都去利用二分查找找到A中小于x的元素数量和。
最后
		for (int i =0; i < n; i++) {
            long cost = Math.min(smaller[i], i - count[instructions[i]] - smaller[i]);
            ret = (ret + cost) % (long) (1e9 + 7);
            count[instructions[i]]++;
        }
利用count记录每个元素出现的次数，所有小于等于x的元素和即为count[instructions[i]] + smaller[i]
```



```java
//1649. Create Sorted Array through Instructions
public class Solution4 {
    long ret = 0;

    long[] count;

    long[] smaller;

    int[] temp;

    public int createSortedArray(int[] instructions) {
        int n = instructions.length;
        temp = new int[n + 1];
        //记录遍历到i时，instructions[i]已经出现的次数
        count = new long[100001];
        smaller = new long[n];
        int[] sorted = Arrays.copyOf(instructions, n);
        divideConque(sorted, instructions, 0, n - 1);
        for (int i =0; i < n; i++) {
            long cost = Math.min(smaller[i], i - count[instructions[i]] - smaller[i]);
            ret = (ret + cost) % (long) (1e9 + 7);
            count[instructions[i]]++;
        }
        return (int) ret;
    }

    private void divideConque(int[] sorted, int[] instructions, int start, int end) {
        if (start >= end) return;
        int mid = start + (end - start) / 2;
        divideConque(sorted, instructions, start, mid);
        divideConque(sorted, instructions, mid + 1, end);
        mergeDivide(sorted, instructions, start, end, mid);
    }

    private void mergeDivide(int[] sorted, int[] instructions, int start, int end, int mid) {
        for (int i = mid + 1; i <= end; i++) {
            // "5" |1 2 3 * 5 5 * 7 8 9|
            //严格小于instructions[i]，所以是求左边界闭区间
            int left = binarySearchL(sorted, instructions[i], start, mid);
            smaller[i] += left - start;
        }
        int index1 = start, index2 = mid + 1, p = 0;
        while (index1 <= mid && index2 <= end) {
            if (sorted[index1] >= sorted[index2]) {
                temp[p] = sorted[index2++];
            } else {
                temp[p] = sorted[index1++];
            }
            p++;
        }
        while (index1 <= mid) {
            temp[p++] = sorted[index1++];
        }
        while (index2 <= end) {
            temp[p++] = sorted[index2++];
        }
        for (int i = 0; i < end - start + 1; i++) {
            sorted[i + start] = temp[i];
        }
    }

    private int binarySearchL(int[] sorted, int instruction, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sorted[mid] >= instruction) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```









## [2426. Number of Pairs Satisfying Inequality](https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/)

```java
//2426. Number of Pairs Satisfying Inequality
public class Solution5 {
    int[] temp;

    long ret = 0;

    public long numberOfPairs(int[] nums1, int[] nums2, int diff) {
        int n = nums1.length;
        temp = new int[n + 1];
        int[] numsDiff = new int[n];
        for (int i = 0; i < n; i++) {
            numsDiff[i] = nums1[i] - nums2[i];
        }
        int[] sorted = Arrays.copyOf(numsDiff, n);
        divideConque(numsDiff, sorted, diff, 0, n - 1);
        return ret;
    }

    private void divideConque(int[] numsDiff, int[] sorted, int diff, int start, int end) {
        if (start >= end) return;
        int mid = start + (end - start) / 2;
        divideConque(numsDiff, sorted, diff, start, mid);
        divideConque(numsDiff, sorted, diff, mid + 1, end);
        mergeSorted(numsDiff, sorted, diff, start, end, mid);
    }

    private void mergeSorted(int[] numsDiff, int[] sorted, int diff, int start, int end, int mid) {
        for (int i = start; i <= mid; i++) {
            //本题取左边界
            int left = binarySearch(sorted, numsDiff[i], diff, mid + 1, end);
            ret += end - left + 1;
        }
        int index1 = start, index2 = mid + 1, p = 0;
        while (index1 <= mid && index2 <= end) {
            if (sorted[index1] <= sorted[index2]) {
                temp[p] = sorted[index1++];
            } else {
                temp[p] = sorted[index2++];
            }
            p++;
        }
        while (index1 <= mid) {
            temp[p++] = sorted[index1++];
        }
        while (index2 <= end) {
            temp[p++] = sorted[index2++];
        }
        for (int i = 0; i < end - start + 1; i++) {
            sorted[i + start] = temp[i];
        }
    }

    private int binarySearch(int[] sorted, int target, int diff, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sorted[mid] >= target - diff) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```









# Combinations

## [77. Combinations](https://leetcode.cn/problems/combinations/)

```
[1 2 3 4 5 6 7]
1 2 3 4 5
1 2 3 4 6
1 2 3 4 7
1 2 3 5 6
1 2 3 5 7
1 2 3 6 7
1 2 4 5 6
...
我们希望可以让子数组以单调增展示
对于i位置的元素，存在combo[i]:[i + 1, i + 1 + n - k]
```

```java
List<List<Integer>> rets = new ArrayList<>();

public List<List<Integer>> combine2(int n, int k) {
    List<Integer> combo = new ArrayList<>();
    for (int i = 1; i <= k; i++) {
        combo.add(i);
    }
    rets.add(combo);
    while (true) {
        int i = k - 1;
        while (i >= 0 && combo.get(i) == i + 1 + n - k) {
            i--;
        }
        if (i < 0) break;
        Integer comboI = combo.get(i);
        comboI++;
        combo.set(i, comboI);
        for (int j = i + 1; j < k; j++) {
            combo.set(j, combo.get(j - 1) + 1);
        }
        rets.add(combo);
    }
    return rets;
}
```

同样DFS也可以解决本题

```java
List<List<Integer>> rets = new ArrayList<>();

public List<List<Integer>> combine(int n, int k) {
    List<Integer> combo = new ArrayList<>();
    DFS(n, k, 1, combo);
    return rets;
}

private void DFS(int n, int k, int index, List<Integer> combo) {
    if (combo.size() >= k) {
        rets.add(combo);
    }
    for (int i = index; i <= n; i++) {
        combo.add(i);
        DFS(n, k, i + 1, combo);
        combo.remove(combo.size() - 1);
    }
}
```











## [1286. Iterator for Combination](https://leetcode.cn/problems/iterator-for-combination/)

```java
//1286. Iterator for Combination
public class CombinationIterator {
    private int[] chars;

    private List<String> ret;

    private int count = 0;

    public CombinationIterator(String characters, int combinationLength) {
        this.chars = new int[26];
        this.ret = new ArrayList<>();
        for (int i = 0; i < characters.length(); i++) {
            char c = characters.charAt(i);
            chars[c - 'a']++;
        }
        DFS(0, combinationLength, new StringBuilder());
    }

    private void DFS(int index, int combinationLength, StringBuilder sb) {
        if (index >= 26) {
            return;
        }
        if (sb.length() >= combinationLength) {
            ret.add(sb.toString());
            return;
        }
        if (chars[index] > 0) {
            chars[index]--;
            sb.append((char) ('a' + index));
            DFS(index, combinationLength, sb);
            chars[index]++;
            sb.deleteCharAt(sb.length() - 1);
            DFS(index + 1, combinationLength, sb);
        } else {
            DFS(index + 1, combinationLength, sb);
        }
    }

    public String next() {
        if (count >= ret.size()) {
            return null;
        } else {
            String s = ret.get(count);
            count++;
            return s;
        }
    }

    public boolean hasNext() {
        if (count >= ret.size()) {
            return false;
        } else {
            return true;
        }
    }
}
```









## [1467. Probability of a Two Boxes Having The Same Number of Distinct Balls](https://leetcode.cn/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/)

```
Good: the number of distributions s.t.(subject to受限于) # colors in box1 = # colors in box2 ->分子
---------------
Total: the number of distributions s.t. # balls in box1 = # balls in box2 ->分母

如果x球有n个，那么我的拿法有(0,n),(1,n-1),(2,n-2)...(n,0)
```

```java
private long equals = 0;

private long count = 0;

private int[] subSetA;

private int[] subSetB;

public double getProbability(int[] balls) {
    subSetA = new int[balls.length];
    subSetB = Arrays.copyOf(balls, balls.length);
    int sum = 0;
    for (int ball : balls) {
        sum += ball;
    }
    DFS(balls, 0, sum / 2, 0);
    return (double) equals / (double) count;
}

private void DFS(int[] balls, int index, int limit, int counted) {
    //当每种颜色的球的所拿数量都确定后
    if (index >= balls.length) {
        //判断两个集合的球总数是否相同
        if (counted == limit) {
            long valid = 1;
            //此时开始求组合数
            for (int i = 0; i < balls.length; i++) {
                //求组合数，在balls[i]个x球中拿subSetA[i]个x球有C[balls[i], subSetA[i]]种可能
                valid *= comb(balls[i], subSetA[i]);
            }
            count += valid;
            //只有两个集合的颜色种数相同时才算是满足题目要求
            if (checkColor(subSetA) == checkColor(subSetB)) {
                equals += valid;
            }
        }
        return;
    }
    //对于每一种颜色的球，我都有这样的拿法：(0,n),(1,n-1),(2,n-2)...(n,0)
    for (int i = 0; i <= balls[index]; i++) {
        subSetA[index] += i;
        subSetB[index] -= i;
        DFS(balls, index + 1, limit, counted + i);
        subSetA[index] -= i;
        subSetB[index] += i;
    }
}

private long comb(int a, int b) {
    long x = 1, y = 1;
    for (int i = 1; i <= b; i++) {
        x *= a + 1 - i;
        y *= i;
    }
    return x / y;
}

private int checkColor(int[] subSet) {
    int count = 0;
    for (int i : subSet) {
        if (i != 0) {
            count++;
        }
    }
    return count;
}
```









## [1643. Kth Smallest Instructions](https://leetcode.cn/problems/kth-smallest-instructions/)

```
因为只能向右或者向下移动，所以一旦题目的输入destination拿到手，我们就可以知道必须向右多少次，以及向下多少次
比如destination是[2,3]，那么就一定由3个H(H代表水平平移，即向右)和2个V(V代表竖直平移，即向下)组成
此时我们假定第一位是H，
即H**** => C(4,2) = 6
那么第一位为V时，
V**** 一定是大于6的
```

本题如果用DFS是会超时的，所以用上述思路，类似二分法

```java
private long comb[][];
public String kthSmallestPath(int[] destination, int k) {
    int V = destination[0];
    int H = destination[1];
    comb = getComb(V + H);
    StringBuilder sb = new StringBuilder();
    findForwards(V, H, k, sb);
    return sb.toString();
}

private void findForwards(int v, int h, int k, StringBuilder sb) {
    int sum = v + h;
    for (int i = 0; i < sum; i++) {
        if (h == 0) {
            sb.append('V');
            v--;
            continue;
        }
        if (v == 0) {
            sb.append('H');
            h--;
            continue;
        }
        long combo = comb[v + h - 1][v];
        if (combo >= (long) k) {
            h--;
            sb.append('H');
        } else {
            v--;
            k -= combo;
            sb.append('V');
        }
    }
}

/*//乘法会溢出
private long comb(int a, int b) {
    long x = 1, y = 1;
    for (int i = 1; i <= b; i++) {
        x *= a + 1 - i;
        y *= i;
    }
    return x / y;
}*/

private long[][] getComb(int n) {
    if (n <= 0) {
        return null;
    }
    long[][] combo = new long[n][n];
    combo[0][0] = 1;
    for (int i = 1; i < n; i++) {
        combo[i][0] = 1;
        for (int j = 1; j <= i; j++) {
            combo[i][j] = combo[i - 1][j - 1] + combo[i - 1][j];
        }
    }
    return combo;
}
```











## [1735. Count Ways to Make Array With Product](https://leetcode.cn/problems/count-ways-to-make-array-with-product/)

```
k balls divided into n groups, allowing empty groups
C(k+n-1,n-1)
可以把空组也视为有效组
Examples(分成三组，相当于插两个板):
2 * 2 * 3 * 3 * 5 * 7 => C(8, 2)
2 * 3 * 3 * 5 * 7 * 2 => C(8, 2)  上下两个不同的排序实际上会产生不同的结果例如下面可以取到5 * 7 * 2但上面怎么都取不到

本题可以转化为求(四个独立概率问题，因此将其相乘即可)
how to divide two 2s into 3 parts? C(2+3-1,3-1)
how to divide two 3s into 3 parts? C(2+3-1,3-1)
how to divide one 5 into 3 parts? C(1+3-1,3-1)
how to divide one 7 into 3 parts? C(1+3-1,3-1)
```

```java
//1735. Count Ways to Make Array With Product
public class Solution5 {
    long[][] combo;

    public int[] waysToFillArray(int[][] queries) {
        combo = getCombo(10016, 15);
        List<Integer> primes = eratosthenes(10001);
        int[] ret = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int n = queries[i][0];
            int k = queries[i][1];
            Map<Integer, Integer> primeGCDs = new HashMap<>();
            for (Integer prime : primes) {
                while (k % prime == 0) {
                    primeGCDs.put(prime, primeGCDs.getOrDefault(prime, 0) + 1);
                    k /= prime;
                }
            }
            if (k != 1) {
                primeGCDs.put(k, primeGCDs.getOrDefault(k, 0) + 1);
            }
            long count = 1;
            Iterator<Map.Entry<Integer, Integer>> it = primeGCDs.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<Integer, Integer> next = it.next();
                Integer value = next.getValue();
                count = count * combo[n + value - 1][value] % (long) (1e9 + 7);
            }
            ret[i] = (int) count;
        }
        return ret;
    }

    private long[][] getCombo(int n, int m) {
        long[][] dp = new long[n][m];
        dp[0][0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= Math.min(i, m - 1); j++) {
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % (long) (1e9 + 7);
            }
        }
        return dp;
    }

    private List<Integer> eratosthenes(int n) {
        int[] q = new int[n + 1];
        List<Integer> primes = new ArrayList<>();
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (q[i] == 1) {
                continue;
            }
            int j = i * 2;
            while (j <= n) {
                q[j] = 1;
                j += i;
            }
        }
        for (int i = 2; i <= n; i++) {
            if (q[i] == 0) {
                primes.add(i);
            }
        }
        return primes;
    }
}
```









## [1866. Number of Ways to Rearrange Sticks With K Sticks Visible](https://leetcode.cn/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/)

```
# Using n numbers to construct a permutation
n! 
123, 132, 213, 231, 312, 321
C(n, 1) * C(n - 1, 1) * ... * C(1, 1)
# Using n numbers to construct a circular permutation or # considered as, fixing the head, running permutation for the rest
在全排列基础上去重，因为一个环中有n种可能，那就是除去n。                 (n - 1)!也可以理解为是固定头元素，剩下的位置全排列
n!/n = (n - 1)!
123 (231, 312), 132 (321, 213)
# Choosing m from n numbers to construct a permutation
A(n, m) = n!/(n - m)!
# Choosing m from n numbers to construct a circular permutation
H(n, m) = n!/(n - m)!/m
# Using n numbers to construct m circular permutations
dp[i][j]: the number of ways that we can use first i numbers to construct j circular permutations
1. if i-th element is for a new circular permutation
dp[i - 1][j - 1]
2. insert the i-th element to the previous j circular permutations
dp[i - 1][j] * (i - 1)

Sterling I:
S[i][j] = S[i - 1][j - 1] + S[i - 1][j] * (i - 1)
```













# Indexing sort

```
indexing sort: 如果确定该数组的元素包含[1,n]，那么排序该数组的时间复杂度可以是O(N)
[2 4 3 5 1]
A[0]是2，而2应该在index=1的位置，所以先交换2和4
[4 2 3 5 1]
4应该在index=3的位置，交换4和5
[5 2 3 4 1]
5应该在index=4的位置，交换5和1
[1 2 3 4 5]
因此可以用O(N)完成对1到n的数进行排序

如果该数组缺失了1到n中的某些数
如[2 4 1 5 2]
  [4 2 1 5 2]
  [5 2 1 4 2]
  [2 2 1 4 5]
  此时应该有循环退出条件，然后依次遍历每一位，遍历到index=2时，调换2和1
  [1 2 2 4 5]
  
为了方便思考，我们可以在数组前加一个0元素，使index=0的元素就是0
```

## [41. First Missing Positive](https://leetcode.cn/problems/first-missing-positive/)

```java
//41. First Missing Positive
public class Solution1 {
    public int firstMissingPositive(int[] nums) {
        int start = 0;
        // Put all non positive numbers to the beginning of the array, and log the number of non positive numbers. Then we can sort the rest of the elements from the index we just logged.
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) {
                int temp = nums[start];
                nums[start] = nums[i];
                nums[i] = temp;
                start++;
            }
        }
        // 1. To avoid array out of bound (nums[i] <= nums.length) 2. To make sure the array continuity (nums[i] != i - start + 1), like 1 should be at index of 'start' 3. To avoid death loop (nums[i] != nums[nums[i] + start - 1])
        for (int i = start; i < nums.length; i++) {
            while (nums[i] <= nums.length - start && nums[i] != i - start + 1 && nums[i] != nums[nums[i] + start - 1]) {
                int index = nums[i] + start - 1;
                int temp = nums[i];
                nums[i] = nums[index];
                nums[index] = temp;
            }
        }
        for (int i = start; i < nums.length; i++) {
            int missing = i - start + 1;
            if (nums[i] != missing) {
                return missing;
            }
        }
        return nums.length - start + 1;
    }
}
```













## [268. Missing Number](https://leetcode.cn/problems/missing-number/)

```java
//268. Missing Number
public class Solution2 {
    public int missingNumber(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] < nums.length && nums[i] != i && nums[i] != nums[nums[i]]) {
                int tempIndex = nums[i];
                int tempVal = nums[i];
                nums[i] = nums[tempIndex];
                nums[tempIndex] = tempVal;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i) {
                return i;
            }
        }
        return nums.length;
    }
}
```











## [442. Find All Duplicates in an Array](https://leetcode.cn/problems/find-all-duplicates-in-an-array/)

```
Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.
Example:
4 3 2 7 8 2 3 1
1 3 2 7 8 2 3 4
1 2 3 7 8 2 3 4
1 2 3 3 8 2 7 4
1 2 3 3 4 2 7 8
1 2 3 4 3 2 7 8 -> final version
Cause array's length is n, and the elements range in [1, n], so use nums[i] as index will never out of bound. After indexing sort, the array will looks ordered except the duplicated elements.
```

```java
//442. Find All Duplicates in an Array
public class Solution3 {
    public List<Integer> findDuplicates(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] <= nums.length && nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]) {
                int tempIdx = nums[i] - 1;
                int tempVal = nums[i];
                nums[i] = nums[tempIdx];
                nums[tempIdx] = tempVal;
            }
        }
        List<Integer> rets = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                rets.add(nums[i]);
            }
        }
        return rets;
    }
}
```







## [448. Find All Numbers Disappeared in an Array](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/)

```
Almost the same as 442. Find All Duplicates in an Array
```

```java
//448. Find All Numbers Disappeared in an Array
public class Solution4 {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] <= nums.length && nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]) {
                int tempIdx = nums[i] - 1;
                int tempVal = nums[i];
                nums[i] = nums[tempIdx];
                nums[tempIdx] = tempVal;
            }
        }
        List<Integer> rets = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                rets.add(i + 1);
            }
        }
        return rets;
    }
}
```







## [645. Set Mismatch](https://leetcode.cn/problems/set-mismatch/)

```java
//645. Set Mismatch
public class Solution5 {
    public int[] findErrorNums(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] <= nums.length && nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]) {
                int tempIdx = nums[i] - 1;
                int tempVal = nums[i];
                nums[i] = nums[tempIdx];
                nums[tempIdx] = tempVal;
            }
        }
        int[] rets = new int[2];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                rets[0] = nums[i];// the duplicated element
                rets[1] = i + 1;// the missing element
            }
        }
        return rets;
    }
}
```









## [2471. Minimum Number of Operations to Sort a Binary Tree by Level](https://leetcode.cn/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/)

```
unorderedVal 7 6 8 5
orderedVal   5 6 7 8
index        0 1 2 3
```

```java
//2471. Minimum Number of Operations to Sort a Binary Tree by Level
public class Solution6 {
    public int minimumOperations(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerLast(root);
        int ret = 0;
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> unordered = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode curNode = deque.pollFirst();
                if (curNode.left != null) {
                    unordered.add(curNode.left.val);
                    deque.offerLast(curNode.left);
                }
                if (curNode.right != null) {
                    unordered.add(curNode.right.val);
                    deque.offerLast(curNode.right);
                }
            }
            List<Integer> ordered = new ArrayList<>(unordered);
            Collections.sort(ordered);
            // Use map's key-val structure to not only store the val, but also the indexes of each level's nodes
            // Example: unorderedVal 7 6 8 5
            //          orderedVal   5 6 7 8
            //          index        0 1 2 3
            Map<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < ordered.size(); i++) {
                map.put(ordered.get(i), i);
            }
            // indexing sort
            for (int i = 0; i < unordered.size(); i++) {
                while (map.get(unordered.get(i)) != i) {
                    Integer changeIdx = map.get(unordered.get(i));
                    Integer tempVal = unordered.get(i);
                    unordered.set(i, unordered.get(changeIdx));
                    unordered.set(changeIdx, tempVal);
                    ret++;
                }
            }
        }
        return ret;
    }
}
```





# Hash & Prefix

## [525. Contiguous Array](https://leetcode.cn/problems/contiguous-array/)

> Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1
> Example: 
> index     0   1 2  3  4 5 6 7  8  9
> array 	      0 1  1  0 1 1 1  0  0
> 	       			|
> 		           -1  1 1 -1 1 1 1 -1 -1
> prefix -> 0 -1 0  1  0 1 2 3  2  1
> See 0 as -1, when prefix = 0, it means the array contains the same number of 0 and 1.
> Then use HashMap to record each prefix, 'key' is the prefix value, 'val' is the smallest index of the array when prefix value = 'key'.
> Iterate the prefix array from the end, if map contains the key of 'prefix[i]', there is a subarray match the rules.

```java
//525. Contiguous Array
public class Solution1 {
    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int[] prefix = new int[nums.length + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + (nums[i - 1] == 0 ? -1 : 1);
        }
        for (int i = 0; i < prefix.length; i++) {
            if (!map.containsKey(prefix[i])) {
                map.put(prefix[i], i);
            }
        }
        int max = 0;
        for (int i = prefix.length - 1; i > 0; i--) {
            if (map.containsKey(prefix[i])) {
                Integer index = map.get(prefix[i]);
                max = Math.max(max, i - index);
            }
        }
        return max;
    }
}
```







## [930. Binary Subarrays With Sum](https://leetcode.cn/problems/binary-subarrays-with-sum/)

> Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.
>
> 1. Construct prefix array.
> 2. Each prefix will store into the map, 'val' is the number of the prefix.

```java
//930. Binary Subarrays With Sum
public class Solution2 {
    public int numSubarraysWithSum(int[] nums, int goal) {
        int[] prefix = new int[nums.length + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }
        Map<Integer, Integer> map = new HashMap<>();
        int ret = 0;
        for (int i = 0; i < prefix.length; i++) {
            if (i != 0) {
                ret += map.getOrDefault(prefix[i] - goal, 0);
            }
            map.put(prefix[i], map.getOrDefault(prefix[i], 0) + 1);
        }
        return ret;
    }
}
```







## [974. Subarray Sums Divisible by K](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

> Tips for getting modulus in Java:
> -5 % 2 = -2...-1
> for getting a positive modulus:
> we can change the formula from i % k -> (i % k + k) % k to get a positive modulus
> -5 % 2 = -2...-1  ->  (-5 % 2 + 2) % 2 = 0...1
>
> if a - b is divisible by c, then a % c = b % c

```java
//974. Subarray Sums Divisible by K
public class Solution3 {
    public int subarraysDivByK(int[] nums, int k) {
        int sum = 0;
        int ret = 0;
        Map<Integer, Integer> map = new HashMap<>();// key: presum % k, val: the number of such prefix sum
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            sum = sum + nums[i];
            int r = (sum % k + k) % k;// for getting a positive modulus
            Integer n = map.getOrDefault(r, 0);
            ret += n;
            map.put(r, n + 1);
        }
        return ret;
    }
}
```





## [1983. Widest Pair of Indices With Equal Range Sum](https://leetcode.cn/problems/widest-pair-of-indices-with-equal-range-sum/)

> presum1[i + 1] - presum1[j] = presum2[i + 1] - presum2[j] -> presum1[i + 1] - presum2[i + 1] = presum1[j] - presum2[j]
> We can use Map to record the earliest presum1[j] - presum2[j] as 'val', and its index as 'key'
> Cause we iterate from 0, so if map.containsKey = false, it means this is the first time presum1[j] - presum2[j] show up

```java
//1983. Widest Pair of Indices With Equal Range Sum
public class Solution4 {
    public int widestPairOfIndices(int[] nums1, int[] nums2) {
        int[] presum1 = new int[nums1.length + 1];
        int[] presum2 = new int[nums2.length + 1];
        for (int i = 1; i < presum1.length; i++) {
            presum1[i] = presum1[i - 1] + nums1[i - 1];
            presum2[i] = presum2[i - 1] + nums2[i - 1];
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        int max = 0;
        for (int i = 1; i < presum1.length; i++) {
            int sumI = presum1[i] - presum2[i];
            if (map.containsKey(sumI)) {
                Integer index = map.get(sumI);
                max = Math.max(max, i - index);
            } else {
                map.put(sumI, i);
            }
        }
        return max;
    }
}
```









## [1371. Find the Longest Substring Containing Vowels in Even Counts](https://leetcode.cn/problems/find-the-longest-substring-containing-vowels-in-even-counts/)

> Given the string s, return the size of the longest substring containing each vowel an even number of times. That is, 'a', 'e', 'i', 'o', and 'u' must appear an even number of times.
>
> We can only use prefix to record whether each vowel element in subarray is an even number
> like '11011' means a, e, o, u appear even number of times, and i appear odd number of times.

```java
//1371. Find the Longest Substring Containing Vowels in Even Counts
public class Solution5 {
    Map<Character, Integer> dict = new HashMap<>();

    public Solution5() {
        this.dict.put('a', 0);
        this.dict.put('e', 1);
        this.dict.put('i', 2);
        this.dict.put('o', 3);
        this.dict.put('u', 4);
    }

    public int findTheLongestSubstring(String s) {
        Map<String, Integer> map = new HashMap<>();
        int[][] prefix = new  int[s.length() + 1][5];
        // '1' means even number times of vowel
        map.put("11111", 0);
        int maxLength = 0;
        for (int i = 1; i < prefix.length; i++) {
            char c = s.charAt(i - 1);
            prefix[i] = Arrays.copyOf(prefix[i - 1], prefix[0].length);
            if (dict.containsKey(c)) {
                Integer index = dict.get(c);
                prefix[i][index]++;
            }
            StringBuilder sb = new StringBuilder();
            for (int x : prefix[i]) {
                sb.append(x % 2 == 0 ? '1' : '0');
            }
            if (map.containsKey(sb.toString())) {
                Integer pre = map.get(sb.toString());
                maxLength = Math.max(maxLength, i - pre);
            } else {
                map.put(sb.toString(), i);
            }
        }
        return maxLength;
    }
}
```







## [1442. Count Triplets That Can Form Two Arrays of Equal XOR](https://leetcode.cn/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)

Given an array of integers arr.

We want to select three indices i, j and k where (0 <= i < j <= k < arr.length).

Let's define a and b as follows:

- `a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]`
- `b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]`

Return *the number of triplets* (`i`, `j` and `k`) Where `a == b`.

> XOR: it outputs true whenever the inputs differ
>
> A ^ A = 0  A XOR itself equals 0
> Case 1:
> Let a == b, then a ^ b = 0
> Case 2:
> a1 ^ a2 ^ a3 ^ a4 ^ a5 = 0
> then {a1 ^ a2} == {a3 ^ a4 ^ a5} and {a1 ^ a2 ^ a3} == {a4 ^ a5} ... {a1 ^ ... ^ ax} == {ax+1 ^ ... ^ an}
> we can just find the i and k that arr[i] ^ ... ^ a[k] == 0, then any of the j in range (i,k] will let a == b
> so, when we find one {i,k}, there are k - i number of ways satisfy a == b

```java
//1442. Count Triplets That Can Form Two Arrays of Equal XOR
public class Solution6 {
    public int countTriplets(int[] arr) {
        int[] prefix = new int[arr.length + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] ^ arr[i - 1];
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        List<Integer> init = new ArrayList<>();
        init.add(0);
        map.put(0, init);
        int count = 0;
        for (int i = 1; i < prefix.length; i++) {
            List<Integer> list = map.getOrDefault(prefix[i] ^ 0, new ArrayList<>());
            for (Integer index : list) {
                count += i - index - 1;
            }
            List<Integer> list2 = map.getOrDefault(prefix[i], new ArrayList<>());
            list2.add(i);
            map.put(prefix[i], list2);
        }
        return count;
    }
}
```







## [1524. Number of Sub-arrays With Odd Sum](https://leetcode.cn/problems/number-of-sub-arrays-with-odd-sum/)

```java
//1524. Number of Sub-arrays With Odd Sum
public class Solution7 {
    public int numOfSubarrays(int[] arr) {
        int[] isOdd = new int[arr.length + 1];
        for (int i = 1; i < isOdd.length; i++) {
            if ((long) (arr[i - 1] + isOdd[i - 1]) % 2 == 0) {
                isOdd[i] = 0;
            } else {
                isOdd[i] = 1;
            }
        }
        int odds = 0;
        int evens = 1;//isOdd[0] = 0, it is a even number
        int ret = 0;
        for (int i = 1; i < isOdd.length; i++) {
            if (isOdd[i] == 0) {
                ret = (ret + odds) % (int) (1e9 + 7);
                evens = (evens + 1) % (int) (1e9 + 7);

            } else {
                ret = (ret + evens) % (int) (1e9 + 7);
                odds = (odds + 1) % (int) (1e9 + 7);
            }
        }
        return ret;
    }
}

```





## [1542. Find Longest Awesome Substring](https://leetcode.cn/problems/find-longest-awesome-substring/)

> Two ways will satisfy the content:
> 1. all characters: count are even
> 2. only one character: count is odd
> 	with perfix[i - 1] is odd/even known, we'd like to know the smallest i
> 	key: odd/even of perfix, value: idx
> 	
>
> Use a String with 10 characters to record the state:
> 0 - even    1 - odd
> like "0001000001" -> this subarray contains odd number of '3' and '9', others are all even.

```java
//1542. Find Longest Awesome Substring
public class Solution8 {
    public int longestAwesome(String s) {
        String[] preState = new String[s.length() + 1];
        Map<String, Integer> map = new HashMap<>();
        map.put("0000000000", 0);
        char[] state = new char[10];
        Arrays.fill(state, '0');
        int ret = 1;
        for (int i = 1; i < preState.length; i++) {
            char c = s.charAt(i - 1);
            char c1 = state[c - '0'];
            state[c - '0'] = (c1 == '0' ? '1' : '0');
            String output = new String(state);
            preState[i] = output;
            if (output.equals("0000000000")) {
                ret = Math.max(ret, i);
                continue;
            }
            for (int j = 0; j < 10; j++) {
                char[] check = new char[10];
                Arrays.fill(check, '0');
                if (state[j] == '0') {
                    check[j] = '1';
                }
                for (int k = 0; k < 10; k++) {
                    if (k == j) continue;
                    if (state[k] == '1') check[k] = '1';
                }
                String key = new String(check);
                if (map.containsKey(key)) {
                    ret = Math.max(ret, i - map.get(key));
                }
            }
            if (!map.containsKey(output)) {
                map.put(output, i);
            }
        }
        return ret;
    }
}
```







## [1590. Make Sum Divisible by P](https://leetcode.cn/problems/make-sum-divisible-by-p/)

> 1 2 3 |4 5 6| 7 8 9 10
> presum[i] - presum[j - 1] = sum[i,j]
> if sum % p = x, we want to find a subsum that (sum - subsum) % p = 0, then subsum % p = x
> Firstly, to get the remainder
> (presum[i] - presum[j - 1]) % p = remainder
> We can put each remainder of prefix[i] divided by p to map, always keep the latest index as value for getting the smallest subarray
> if p = 9, remainder = 4,  58
>    p = 9, r = 5, 41
>    p = 9, r = 1, 19
>    p = 9, r = 3, 39
>    p = 9, r = 8, 26
> if p = 6, remainder = 4, 28
>    p = 6, r = 3, 15
>    p = 6, r = 5, 11
>
> As above, we can find that if r >= remainder, the 'target remainder' is r - remainder
> if r < remainder, then 'target remainder' is p - (remainder - r)

```java
//1590. Make Sum Divisible by P
public class Solution9 {
    public int minSubarray(int[] nums, int p) {
        // use int may cause out of bound, example: [1000000000,1000000000,1000000000], p = 3
        long[] prefix = new long[nums.length + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }
        int remainder = (int) (prefix[prefix.length - 1] % p);
        // if the sum of the array's value is divisible by p, then no elements should be removed, so return 0
        if (remainder == 0) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        int ret = Integer.MAX_VALUE;
        for (int i = 1; i < prefix.length; i++) {
            int r = (int) (prefix[i] % p);
            int target = 0;
            if (r >= remainder) {
                target = r - remainder;
            } else {
                target = p - (remainder - r);
            }
            if (map.containsKey(target)) {
                ret = Math.min(ret, i - map.get(target));
            }
            map.put(r, i);
        }
        // if ret == num.length, it means that for satisfying the content, all the array elements should be removed, it is impossible, so return -1
        return ret != nums.length ? ret : -1;
    }
}
```









## [1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)

> 1. Get the prefix of the array, and put the prefix to Map(val = value of prefix, key = index)
> 2. Get the suffix of the array, each time check if map contains the key of 'x - suffix[i]'

```java
//1658. Minimum Operations to Reduce X to Zero
public class Solution10 {
    public int minOperations(int[] nums, int x) {
        int[] prefix = new int[nums.length + 1];
        int[] suffix = new int[nums.length + 1];
        Map<Integer, Integer> leftPart = new HashMap<>();
        leftPart.put(0, 0);
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
            leftPart.put(prefix[i], i);
        }
        int min = Integer.MAX_VALUE;
        if (leftPart.containsKey(x)) {
            min = Math.min(min, leftPart.get(x));
        }
        for (int i = 1; i < suffix.length; i++) {
            suffix[i] = suffix[i - 1] + nums[nums.length - i];
            if (leftPart.containsKey(x - suffix[i])) {
                Integer left = leftPart.get(x - suffix[i]);
                if (i + left <= nums.length) {
                    min = Math.min(min, i + left);
                }
            }
        }
        return min == Integer.MAX_VALUE ? -1 : min;
    }
}
```







## [1915. Number of Wonderful Substrings](https://leetcode.com/problems/number-of-wonderful-substrings/)

> All the combination that satisfy the content:
> 1. all the letters appear even number of times
> 2. one letter appear odd number of times, others appear even number of times
> when we get a new state, first check if map contains the state equals to itself, for example:
> '0100100000' -> to find '0100100000' itself, equals to get the substring full with even number of times letters
> Then, iterate the state, let each possible letter appear odd number of times, to check if map contains the key

```java
public long wonderfulSubstrings(String word) {
    char[] prestate = new char[10];
    Arrays.fill(prestate, '0');
    Map<String, Integer> map = new HashMap<>();
    map.put(new String(prestate), 1);
    long ret = 0;
    for (int i = 0; i < word.length(); i++) {
        char c = word.charAt(i);
        prestate[c - 'a'] = (prestate[c - 'a'] == '0' ? '1' : '0');
        String output = new String(prestate);
        if (map.containsKey(output)) {
            ret += map.get(output);
        }
        // let each possible letter appear odd number of times
        for (int j = 0; j < 10; j++) {
            char[] match = new char[10];
            Arrays.fill(match, '0');
            if (prestate[j] == '0') {
                match[j] = '1';
            }
            for (int k = 0; k < 10; k++) {
                if (k == j) continue;
                if (prestate[k] == '1') {
                    match[k] = '1';
                }
            }
            String matchString = new String(match);
            if (map.containsKey(matchString)) {
                ret += map.get(matchString);
            }
        }
        map.put(output, map.getOrDefault(output, 0) + 1);
    }
    return ret;
}
```









## [2025. Maximum Number of Ways to Partition an Array](https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/)

> Using prefix and suffix to solve the problem.
> Example:
> Array:  2 -1 2
> prefix: 2  1 3
> suffix: 3  1 2
>
> We can change the value of each element in Array to 'k' by loop, for examples:
> [x] k [x x x x x x]
> [x x] k [x x x x x]
> [x x x] k [x x x x]
> [x x x x] k [x x x]
> ...
> [x x x x x x] k [x]
> 1. Iterate array from index = 0 to index = array.length - 2 (1 <= pivot < n, so the last element that can be iterated is index = array.length - 2), using map to record prefix.
> 2. Iterate array from index = array.length -1 to index = 1, using map to record suffix.

```java
public int waysToPartition(int[] nums, int k) {
    long[] prefix = new long[nums.length];
    long[] suffix = new long[nums.length];
    Map<Long, Integer> prefixMap = new HashMap<>();
    Map<Long, Integer> suffixMap = new HashMap<>();
    prefix[0] = nums[0];
    for (int i = 1; i < prefix.length; i++) {
        prefix[i] = prefix[i - 1] + nums[i];
    }
    suffix[suffix.length - 1] = nums[nums.length - 1];
    for (int i = suffix.length - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] + nums[i];
    }
    int[] ret = new int[nums.length];
    for (int i = 0; i < prefix.length; i++) {
        long newSum = prefix[prefix.length - 1] - nums[i] + k;
        if (newSum % 2 == 0) {
            ret[i] += prefixMap.getOrDefault(newSum / 2, 0);
        }
        prefixMap.put(prefix[i], prefixMap.getOrDefault(prefix[i], 0) + 1);
    }
    for (int i = suffix.length - 1; i >= 0; i--) {
        long newSum = prefix[prefix.length - 1] - nums[i] + k;
        if (newSum % 2 == 0) {
            ret[i] += suffixMap.getOrDefault(newSum / 2, 0);
        }
        suffixMap.put(suffix[i], suffixMap.getOrDefault(suffix[i], 0) + 1);
    }
    int noChange = 0;
    for (int i = 0; i < prefix.length - 1; i++) {
        if (prefix[prefix.length - 1] % 2 != 0) break;
        if (prefix[i] == prefix[prefix.length - 1] / 2) {
            noChange++;
        }
    }
    int max = 0;
    for (int i : ret) {
        max = Math.max(max, i);
    }
    return Math.max(max, noChange);
}
```









## [2488. Count Subarrays With Median K](https://leetcode.com/problems/count-subarrays-with-median-k/)

> The question can be considered as getting all the subarrays in array, the subarray should firstly contain k, and then the number of elements that greater than k minus the number of elements smaller than k should be less equal than 1 and greater equal than 0 (0<= k_larger - k_smaller <= 1). 
> **Explain**:
> If k_larger - k_smaller = 0, the length of the subarray is a odd number, so k is definitly the median. If k_larger - k_smaller = 1, then the length of the subarray is an even number, and k will be the left median that could satisfy the content.
>
> Using **prefix** to record the number of elements greater than k minus the number of elements smaller than k.

```java
public int countSubarrays(int[] nums, int k) {
    // the prefix of the number of elements greater than k minus the number of elements smaller than k
    int[] prefix = new int[nums.length];
    int mark = -1;
    if (nums[0] == k) {
        mark = 0;
        prefix[0] = 0;
    } else {
        prefix[0] = nums[0] > k ? 1 : -1;
    }
    for (int i = 1; i < prefix.length; i++) {
        if (nums[i] == k) {
            mark = i;
            prefix[i] = prefix[i - 1];
        } else {
            prefix[i] = prefix[i - 1] + (nums[i] > k ? 1 : -1);
        }
    }
    if (mark == -1) return 0;
    // key: prefix of the number of elements greater than k minus the number of elements smaller than k
    // value: number of such prefix
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 1);
    int ret = 0;
    for (int i = 0; i < prefix.length; i++) {
        if (i < mark) {
            map.put(prefix[i], map.getOrDefault(prefix[i], 0) + 1);
        } else {
            ret += map.getOrDefault(prefix[i] - 1, 0) + map.getOrDefault(prefix[i], 0);
        }
    }
    return ret;
}
```









# Bit Manipulation

## XOR

```
XOR: it outputs true whenever the inputs differ

A ^ A = 0  A XOR itself equals 0
Case 1:
Let a == b, then a ^ b = 0
Case 2:
a1 ^ a2 ^ a3 ^ a4 ^ a5 = 0
then {a1 ^ a2} == {a3 ^ a4 ^ a5} and {a1 ^ a2 ^ a3} == {a4 ^ a5} ... {a1 ^ ... ^ ax} == {ax+1 ^ ... ^ an}

XOR_sum: xor_sum[i:k] = pre_xor_sum[k] ^ pre_xor_sum[i - 1]

```



# Bit Masks & DP

## [691. Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word/)

```java
    // since target <= 15, we can assume that the result cannot exceed 15, INF is like a maximum limit here, it can be any number greater than 15
    static int N = 15, INF = 20;
    static int[] f = new int[1 << N];
    public int minStickers(String[] ss, String t) {
        int m = t.length(), mask = 1 << m;
        Arrays.fill(f, INF);
        f[0] = 0;
        for (int s = 0; s < mask; s++) {
            if (f[s] == INF) continue;
            // start from state 0, try different stickers, after each iteration, we can get an optimum states array depends on f[ns] = Math.min(f[ns], f[s] + 1);
            for (String str : ss) {
                int ns = s, len = str.length();
                for (int i = 0; i < len; i++) {
                    int c = str.charAt(i) - 'a';
                    for (int j = 0; j < m; j++) {
                        // do right shift on ns, ns = s, and is the current iteration's states, check if the j th state is already satisfied
                        if (t.charAt(j) - 'a' == c && (((ns >> j) & 1) == 0)) {
                            // if satisfied, change the j th state to satisified
                            ns |= (1 << j);
                            break;
                        }
                    }
                }
                f[ns] = Math.min(f[ns], f[s] + 1);
            }
        }
        return f[mask - 1] == INF ? -1 : f[mask - 1];
    }
```



## [1125. Smallest Sufficient Team](https://leetcode.com/problems/smallest-sufficient-team/)

```java
int max_people;

public int[] smallestSufficientTeam(String[] req_skills, List<List<String>> people) {
    int skillLen = req_skills.length;
    max_people = skillLen + 1;
    int[] states = new int[1 << skillLen];
    List<Set<Integer>> chonsen_people_sets = new ArrayList<>(Collections.nCopies((1 << skillLen), new HashSet<>()));
    Arrays.fill(states, max_people);
    states[0] = 0;
    for (int s = 0; s < states.length; s++) {
        // no way to here, contiune to next
        if (states[s] == max_people) continue;
        Set<Integer> current_people_set = chonsen_people_sets.get(s);
        for (int p = 0; p < people.size(); p++) {
            // if the people is already chosen, continue
            if (current_people_set.contains(p)) continue;
            Set<Integer> next_people_set = new HashSet<>(current_people_set);
            int temp_s = s;
            List<String> p_skills = people.get(p);
            for (int i = 0; i < p_skills.size(); i++) {
                String p_skill = p_skills.get(i);
                for (int j = 0; j < skillLen; j++) {
                    if (req_skills[j].equals(p_skill) && ((temp_s >> j) & 1) == 0) {
                        temp_s |= (1 << j);
                        break;
                    }
                }
            }
            // dynamically get the minimum number of people
            if (temp_s != s && states[temp_s] > states[s] + 1) {
                states[temp_s] = states[s] + 1;
                next_people_set.add(p);
                chonsen_people_sets.set(temp_s, next_people_set);
            }
        }
    }
    Set<Integer> res_set = chonsen_people_sets.get((1 << skillLen) - 1);
    int length = res_set.size();
    int[] res = new int[length];
    int index = 0;
    for (Integer p : res_set) {
        res[index++] = p;
    }
    return res;
}
```



## [1349. Maximum Students Taking Exam](https://leetcode.com/problems/maximum-students-taking-exam/)

```java
//1349. Maximum Students Taking Exam
public class MaximumStudentsTakingExam {
    char[][] seats_global;
    int m, n;

    public int maxStudents(char[][] seats) {
        // The current row's elements only affected by the previous row's elements, and the current row's elements
        // dp[i][p] means maximum number of students that can take the exam together without any cheating being possible by the i-th row, and we choose pattern p at the i-th row
        // dp[i][p] = max(dp[i-1][t]), t range in [0, maximum pattern] (for t = 0, 1, ..., 2^N), not conflicts with p
        seats_global = seats;
        m = seats.length;
        n = seats[0].length;
        int[] dp = new int[(1 << n)];
        for (int p = 0; p < (1 << n); p++) {
            if (isSameRowNoConflicts(p, 0)) {
                dp[p] = countSeats(p);
            }
        }
        for (int row = 1; row < m; row++) {
            int[] dp_prev = Arrays.copyOf(dp, dp.length);
            Arrays.fill(dp, 0);
            for (int cur_state = 0; cur_state < (1 << n); cur_state++) {
                // check if in current row has conflicts, for example: 11001, the 1-st and 2-nd seats have conflicts
                if (!isSameRowNoConflicts(cur_state, row)) continue;
                // update dp[cur_state]
                for (int prev_state = 0; prev_state < (1 << n); prev_state++) {
                    // check if in previous row has conflicts
                    if (!isSameRowNoConflicts(prev_state, row - 1)) continue;
                    /*
                        check if two cross rows have conflicts, for example:
                        1 0 0
                        0 1 0
                        1-st row's 1-st element have conflict with 2-nd row's 2-nd element
                    */
                    if (!isPrevRowNoConflicts(cur_state, prev_state)) continue;
                    dp[cur_state] = Math.max(dp_prev[prev_state] + countSeats(cur_state), dp[cur_state]);
                }
            }
        }
        int res = 0;
        for (int p = 0; p < (1 << n); p++) {
            res = Math.max(res, dp[p]);
        }
        return res;
    }

    /*count the number of 1 of a binary number*/
    private int countSeats(int cur_state) {
        int res = 0;
        while (cur_state > 0) {
            res += cur_state % 2;
            cur_state /= 2;
        }
        return res;
    }

    private boolean isPrevRowNoConflicts(int cur_state, int prev_state) {
        int[] temp_cur = new int[n];
        int[] temp_prev = new int[n];
        for (int i = 0; i < n; i++) {
            temp_cur[i] = cur_state % 2;
            temp_prev[i] = prev_state % 2;
            cur_state /= 2;
            prev_state /= 2;
        }
        for (int i = 0; i < n; i++) {
            // if current seat and top-left seat are both occupied
            if (temp_cur[i] == 1 && i - 1 >= 0 && temp_prev[i - 1] == 1) {
                return false;
            }
            // if current seat and top-right seat are both occupied
            if (temp_cur[i] == 1 && i + 1 < n && temp_prev[i + 1] == 1) {
                return false;
            }
        }
        return true;
    }

    private boolean isSameRowNoConflicts(int cur_state, int row) {
        int[] temp = new int[n];
        for (int i = 0; i < n; i++) {
            temp[i] = cur_state % 2;
            cur_state /= 2;
        }
        for (int i = 0; i < n; i++) {
            // if it is a broken seat, we can not set the pattern to 1 means someone seat there
            if (temp[i] == 1 && seats_global[row][i] == '#') {
                return false;
            }
            // It's left position cannot seat
            if (temp[i] == 1 && i - 1 >= 0 && temp[i - 1] == 1) {
                return false;
            }
        }
        return true;
    }
}
```



## [1434. Number of Ways to Wear Different Hats to Each Other](https://leetcode.com/problems/number-of-ways-to-wear-different-hats-to-each-other/)

```
Way1:
// state: binary type, i-th bit represents if the i-th hat has been taken
// dp[state]: number of ways for this case
for people in [0, n]:
	for state in [00...00, 11...11]:
		for hat in HatsForThisPeople:
			if hat has been taken in state: continue
			dp_new[state+hat] += dp[state]
ans = sum(dp[state]) for those states contain n bit 1
// But the time complexity is big

Way2:
// i-th bit represents if the i-th people has taken a hat
// dp[state]: number of ways for this case
for hats in [1, 40]:
	for state in [00...00, 11...11]:
		for people in PersonForThisHat:
			if people has taken hat in this state: continue
			dp_new[state+people] += dp[state]
ans = dp[state] which state full filled with 1
```

```java
// 1434. Number of Ways to Wear Different Hats to Each Other
public class NumberOfWays2WearDifferentHats2EachOther {
    int max_nHats = 40;
    int n;
    List<List<Integer>> hats;

    public int numberWays(List<List<Integer>> hats) {
        // i-th bit represents if the i-th people has taken a hat
        // dp[state]: number of ways for this case
        n = hats.size();
        int nState = 1 << n;
        long[] dp = new long[nState];
        // no one take a hat is also one way
        dp[0] = 1;
        Map<Integer, Set<Integer>> personForThisHat = new HashMap<>();
        for (int p = 0; p < n; p++) {
            List<Integer> hatsForThisPerson = hats.get(p);
            for (Integer hat : hatsForThisPerson) {
                Set<Integer> person = personForThisHat.getOrDefault(hat, new HashSet<>());
                if (!person.contains(p)) person.add(p);
                personForThisHat.put(hat, person);
            }
        }
        for (int hat = 1; hat <= 40; hat++) {
            long[] dp_new = Arrays.copyOf(dp, dp.length);
            for (int state = 0; state < nState; state++) {
                Set<Integer> person = personForThisHat.getOrDefault(hat, new HashSet<>());
                for (Integer p : person) {
                    if (hasHat(state, p)) continue;
                    dp_new[state + (1 << p)] = (dp_new[state + (1 << p)] + dp[state]) % (long) (1e9 + 7);
                }
            }
            dp = dp_new;
        }
        return (int) dp[nState - 1];
    }

    private boolean hasHat(int state, int p) {
        // bitwise shift operation
        return ((state >> p) & 1) == 1;
    }
}
```



# BinarySearch Template

```java
//若数组中有多个x，则二分查找寻找i的左边界闭区间，开区间请改成sorted[mid] > target + l即可(但此时的开区间是往大的找的，即返回值是x+1)
private int binarySearchLeft(long[] sorted, long target, int start, int end) {
    int left = start, right = end;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (sorted[mid] >= target) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

//二分查找寻找x右边界，开区间请改成sorted[mid] < target + u即可(同理返回值是x+1)
private int binarySearchRight(long[] sorted, long target, int start, int end) {
    int left = start, right = end;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (sorted[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}

//[start, target)
private int binarySearchRight2(long[] sorted, lont target, int start, int end) {
    int left = start, right = end;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (sorted[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}
```







# Stable Matching

```typescript
import "./styles.css";
import axios from 'axios';
import React, { useState, useEffect } from 'react'

interface Todo {
  userId: number;
  id: number;
  title: string;
  completed: boolean;
}

export default function App() {
  const [filteredTodos, setFilteredTodos] = useState<Todo[]>([]);
  const [todos, setTodos] = useState<Todo[]>([]);
  const [searchField, setSearchField] = useState<string>("");
  console.log(searchField);
  useEffect(() => {
    axios.get('https://jsonplaceholder.typicode.com/todos?_limit=10&_sort=id&_order=desc')
      .then(response => {
        setTodos(response.data);
      })
      .catch(error => {
        console.log(error);
      });
  }, []);

  const searchFieldHandler = (event:React.ChangeEvent<HTMLInputElement>) => {
    setSearchField(event.target.value.toLowerCase())
  }

  useEffect(() => {
    const newFilteredTodos = todos.filter((todo) => {return todo.title.toLowerCase().includes(searchField)})
    setFilteredTodos(newFilteredTodos)
  }, [todos, searchField])
  
  return (
    <div className="App">
      <input type="search" onChange={searchFieldHandler}/>
      <ul>
        {filteredTodos.map((filteredTodo) => (
          <li key={filteredTodo.id}>
            <ul>
              <li>id:{filteredTodo.id}</li>
              <li>UserId:{filteredTodo.userId}</li>
              <li>title:{filteredTodo.title}</li>
              <li>completed:{filteredTodo.completed ? "yes" : "no"}</li>
            </ul>
          </li>
        ))}
      </ul>
      
    </div>
  );
}

```





















# LeetCode Weekly Contests

## 308

### [6160. 和有限的最长子序列](https://leetcode.cn/problems/longest-subsequence-with-limited-sum/)

```java
public int[] answerQueries(int[] nums, int[] queries) {
    Arrays.sort(nums);
    int[] ans = new int[queries.length];
    for (int i = 0; i < queries.length; i++) {
        int sum = 0;
        int index = 0;
        while (index < nums.length && sum <= queries[i]) {
            sum += nums[index];
            index++;
        }
        if (sum <= queries[i]) {
            ans[i] = index;
        } else {
            ans[i] = index - 1;
        }
    }
    return ans;
}
```

### [6161. 从字符串中移除星号](https://leetcode.cn/problems/removing-stars-from-a-string/)

```java
public String removeStars(String s) {
    StringBuilder sb = new StringBuilder(s);
    int index = 0;
    while (index < sb.length()) {
        if (sb.charAt(index) == '*') {
            sb.delete(index - 1, index + 1);
            index -= 2;
            if (index < 0) {
                index = 0;
            }
        }
        index++;
    }
    return sb.toString();
}
```

### [6162. 收集垃圾的最少总时间](https://leetcode.cn/problems/minimum-amount-of-time-to-collect-garbage/)

```java
public int garbageCollection(String[] garbage, int[] travel) {
    int[] consume = new int[3];
    int m = 0, p = 0, g = 0;
    int distant = 0;
    int index = 0;
    for (String each : garbage) {
        distant += (index == 0 ? 0 : travel[index - 1]);
        for (int i = 0; i < each.length(); i++) {
            char c = each.charAt(i);
            if (c == 'M') {
                consume[0]++;
                m = distant;
            }
            if (c == 'P') {
                consume[1]++;
                p = distant;
            }
            if (c == 'G') {
                consume[2]++;
                g = distant;
            }
        }
        index++;
    }
    return consume[0] + consume[1] + consume[2] + m + p + g;
}
```

### [6163. 给定条件下构造矩阵](https://leetcode.cn/problems/build-a-matrix-with-conditions/)

看到题目我就想到了[0207. 课程表](https://leetcode.cn/problems/course-schedule/)，利用入度表来把所有的数字进行一个排序

```java
public int[][] buildMatrix(int k, int[][] rowConditions, int[][] colConditions) {
    int[] rowInDegree = new int[k + 1];
    int[] colInDegree = new int[k + 1];
    Map<Integer, List<Integer>> mapRow = new HashMap<>();
    Map<Integer, List<Integer>> mapCol = new HashMap<>();
    for (int i = 0; i < rowConditions.length; i++) {
        rowInDegree[rowConditions[i][0]]++;
        Integer preNum = rowConditions[i][1];
        Integer curNum = rowConditions[i][0];
        List<Integer> list = mapRow.getOrDefault(preNum, new ArrayList<>());
        list.add(curNum);
        mapRow.put(preNum, list);
    }
    for (int i = 0; i < colConditions.length; i++) {
        colInDegree[colConditions[i][0]]++;
        Integer preNum = colConditions[i][1];
        Integer curNum = colConditions[i][0];
        List<Integer> list = mapCol.getOrDefault(preNum, new ArrayList<>());
        list.add(curNum);
        mapCol.put(preNum, list);
    }
    Deque<Integer> dequeRow = new LinkedList<>();
    for (int i = 1; i < rowInDegree.length; i++) {
        if (rowInDegree[i] == 0) {
            dequeRow.addLast(i);
        }
    }
    int[] rowOrder = new int[k + 1];
    int orderR = 1;
    while (!dequeRow.isEmpty()) {
        Integer pre = dequeRow.pollFirst();
        rowOrder[pre] = orderR;
        orderR++;
        List<Integer> curList = mapRow.containsKey(pre) ? mapRow.get(pre) : new ArrayList<>();
        for (Integer cNum : curList) {
            if (--rowInDegree[cNum] == 0) {
                dequeRow.addLast(cNum);
            }
        }
    }
    for (int i : rowInDegree) {
        if (i != 0) {
            return new int[1][0];
        }
    }
    Deque<Integer> dequeCol = new LinkedList<>();
    for (int i = 1; i < colInDegree.length; i++) {
        if (colInDegree[i] == 0) {
            dequeCol.addLast(i);
        }
    }
    int[] colOrder = new int[k + 1];
    int orderC = 1;
    while (!dequeCol.isEmpty()) {
        Integer pre = dequeCol.pollFirst();
        colOrder[pre] = orderC;
        orderC++;
        List<Integer> curList = mapCol.containsKey(pre) ? mapCol.get(pre) : new ArrayList<>();
        for (Integer cNum : curList) {
            if (--colInDegree[cNum] == 0) {
                dequeCol.addLast(cNum);
            }
        }
    }
    for (int i : colInDegree) {
        if (i != 0) {
            return new int[1][0];
        }
    }
    int[][] ans = new int[k][k];
    for (int i = 1; i <= k; i++) {
        int rowIndex, colIndex;
        if (rowOrder[i] != 0) {
            rowIndex = (k - 1) - (rowOrder[i] - 1);
        } else {
            rowIndex = ++orderR - 1;
        }
        if (colOrder[i] != 0) {
            colIndex = (k - 1) - (colOrder[i] - 1);
        } else {
            colIndex = ++orderC - 1;
        }
        ans[rowIndex][colIndex] = i;
    }
    return ans;
}
```









## 309

### [6167. 检查相同字母间的距离](https://leetcode.cn/problems/check-distances-between-same-letters/)

```java
public boolean checkDistances(String s, int[] distance) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        if (!map.containsKey(c)) {
            map.put(c, i);
        } else {
            Integer first = map.get(c);
            int index = c - 'a';
            if (distance[index] != i - first - 1) {
                return false;
            }
        }
    }
    return true;
}
```









### [6168. 恰好移动 k 步到达某一位置的方法数目](https://leetcode.cn/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/)

```java
int ways = 0;
public int numberOfWays(int startPos, int endPos, int k) {
    checkWays(startPos, endPos, k);
    return ways;
}

private void checkWays(int curPos, int endPos, int k) {
    if (curPos == endPos && k == 0) {
        ways++;
    }
    if (Math.abs(endPos - curPos) > k) {
        return;
    }
    checkWays(curPos + 1, endPos, k - 1);
    checkWays(curPos - 1, endPos, k - 1);
}
```

首先尝试了递归，但是超时，本题使用递归还要加入记忆化搜索，具体想法就是利用map以==当前curPos和剩余步数k==组成标签的key，存入value为在此时的组合数

```java
int MOD = (int) 1e9 + 7;
Map<Integer, Long> map = new HashMap<>();
public int numberOfWays(int startPos, int endPos, int k) {
    return (int) checkWays(startPos, endPos, k);
}

private long checkWays(int curPos, int endPos, int k) {
    if (curPos == endPos && k == 0) {
        return 1;
    }
    if (Math.abs(endPos - curPos) > k) {
        return 0;
    }
    int key = curPos * 1000 + k;//因为k<=1000，所以利用curPos*1000一定可以生成不同的key来区分不同的curPos+k的组合
    if (map.containsKey(key)) {
        return map.get(key);
    }
    long value = (checkWays(curPos + 1, endPos, k - 1) + checkWays(curPos - 1, endPos, k - 1)) % MOD;
    map.put(key, value);
    return value;
}
```

加入记忆化搜索之后，该题将少走很多步骤





### [6169. 最长优雅子数组](https://leetcode.cn/problems/longest-nice-subarray/)

```java
int[] cnt = new int[32];
public int longestNiceSubarray(int[] nums) {
    int left = 0, right = 0, max = 1;
    while (right < nums.length) {
        int num = nums[right];
        String s = Integer.toBinaryString(num);
        if (checkCnt(s)) {
            for (int i = s.length() - 1; i >= 0; i--) {
                if (s.charAt(i) == '1') {
                    cnt[s.length() - 1 - i] = 1;
                }
            }
            max = Math.max(max, right - left + 1);
            right++;
        } else {
            while (!checkCnt(s)) {
                int num1 = nums[left];
                String s1 = Integer.toBinaryString(num1);
                for (int i = s1.length() - 1; i >= 0; i--) {
                    if (s1.charAt(i) == '1') {
                        cnt[s1.length() - 1 - i] = 0;
                    }
                }
                left++;
            }
        }
    }
    return max;
}

private Boolean checkCnt(String s) {
    for (int i = s.length() - 1; i >= 0; i--) {
        if (s.charAt(i) == '1' && cnt[s.length() - 1 - i] == 1) {
            return false;
        }
    }
    return true;
}
```

该题需要吸收一下与&运算：

<img src="https://raw.githubusercontent.com/Prom1s1ngYoung/cloudImg/main/leetcode/image-20220904184909480.png" alt="image-20220904184909480" style="zoom:50%;" />

如果想要两个数做与运算之后为0，那么就是说明在二进制状态下，每一位都只能有至多一个'1'





### [6170. 会议室 III](https://leetcode.cn/problems/meeting-rooms-iii/)

```java
public int mostBooked(int n, int[][] meetings) {
    int[] useTimes = new int[n];
    int max = 0;
    int maxNumber = 0;
    PriorityQueue<Integer> zoomQueue = new PriorityQueue<>(100, new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o1 - o2;
        }
    });
    Arrays.sort(meetings, new Comparator<int[]>(){
        @Override
        public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
                return o1[1] - o2[1];
            }
            return o1[0] - o2[0];
        }
    });
    PriorityQueue<int[]> meetingQueue = new PriorityQueue<>(100, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0] - o2[0];
        }
    });
    for (int i = 0; i < n; i++) {
        zoomQueue.offer(i);
    }
    for (int i = 0; i < meetings.length; i++) {
        while (!meetingQueue.isEmpty()) {
            int startTime = meetings[i][0];
            int[] peek = meetingQueue.peek();
            if (peek[0] > startTime) {
                break;
            }
            zoomQueue.offer(peek[1]);
            meetingQueue.poll();
        }
        if (zoomQueue.isEmpty()) {
            int[] poll = meetingQueue.poll();
            if (poll[0] > meetings[i][0]) {
                meetings[i][1] = meetings[i][1] - meetings[i][0] + poll[0];
            }
            zoomQueue.offer(poll[1]);
        }
        Integer zoomNumber = zoomQueue.poll();
        int[] meeting = new int[2];
        meeting[0] = meetings[i][1];
        meeting[1] = zoomNumber;
        useTimes[zoomNumber]++;
        if (max == useTimes[zoomNumber]) {
            maxNumber = Math.min(maxNumber, zoomNumber);
        } else {
            maxNumber = max > useTimes[zoomNumber] ? maxNumber : zoomNumber;
        }
        max = Math.max(max, useTimes[zoomNumber]);
        meetingQueue.offer(meeting);
    }
    return maxNumber;
}
```

该解法最后两个用例不通过，原因是爆int了，要把meetings优先队列的存储方式换成Pair<Long, Interger>

参考一下别人的写法：

```java
public int mostBooked2(int n, int[][] meetings) {
    int[] cnt = new int[n];
    PriorityQueue<Integer> idle = new PriorityQueue<Integer>((a, b) -> Integer.compare(a, b));
    for (int i = 0; i < n; ++i) idle.offer(i);
    PriorityQueue<Pair<Long, Integer>> using = new PriorityQueue<Pair<Long, Integer>>((a, b) -> !Objects.equals(a.getKey(), b.getKey()) ? Long.compare(a.getKey(), b.getKey()) : Integer.compare(a.getValue(), b.getValue()));
    Arrays.sort(meetings, (a, b) -> Integer.compare(a[0], b[0]));
    for (int[] m : meetings) {
        long st = m[0], end = m[1];
        while (!using.isEmpty() && using.peek().getKey() <= st) {
            idle.offer(using.poll().getValue()); // 维护在 st 时刻空闲的会议室
        }
        int id;
        if (idle.isEmpty()) {
            Pair<Long, Integer> p = using.poll();// 没有可用的会议室，那么弹出一个最早结束的会议室
            end += p.getKey() - st; // 更新当前会议的结束时间
            id = p.getValue();
        } else id = idle.poll();
        ++cnt[id];
        using.offer(new Pair<>(end, id)); // 使用一个会议室
    }
    int ans = 0;
    for (int i = 0; i < n; ++i) if (cnt[i] > cnt[ans]) ans = i;
    return ans;
}
```









## 310

### [6176. 出现最频繁的偶数元素](https://leetcode.cn/problems/most-frequent-even-element/)

```java
public int mostFrequentEven(int[] nums) {
    Map<Integer, Integer> map = new HashMap<>();
    int max = 0;
    int maxN = -1;
    for (int num : nums) {
        if (num % 2 == 0) {
            map.put(num, map.getOrDefault(num, 0) + 1);
            Integer amount = map.get(num);
            if (amount > max) {
                max = amount;
                maxN = num;
            } else if (amount == max) {
                maxN = Math.min(maxN, num);
            }
        }
    }
    return maxN;
}
```



### [6177. 子字符串的最优划分](https://leetcode.cn/problems/optimal-partition-of-string/)

```java
public int partitionString(String s) {
    int left = 0, right = 0;
    int spiltN = 0;
    Set<Character> set = new HashSet<>();
    while (right < s.length()) {
        if (!set.contains(s.charAt(right))) {
            set.add(s.charAt(right));
            right++;
        }else {
            set = new HashSet<>();
            spiltN++;
        }
    }
    if (!set.isEmpty()) spiltN++;
    return spiltN;
}
```



### [6178. 将区间分为最少组数](https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/)

```java
public int minGroups(int[][] intervals) {
    Arrays.sort(intervals, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            if (o1[0] == o2[0]) {
                return o1[1] -o2[1];
            }
            return o1[0] - o2[0];
        }
    });
    PriorityQueue<Integer> queue = new PriorityQueue<>();
    for (int[] interval : intervals) {
        int end = interval[1];
        if (!queue.isEmpty() && queue.peek() < interval[0]) queue.poll();
        queue.offer(end);
    }
    return queue.size();
}
```

首先将输入数组按照start的时间先后进行排序，本题我们利用最小堆来维护多个**组**的end，存入最小堆中的对应实体就是一个个组的右区间，这些值会被动态更新，堆的大小实际就是当前已经划分出来的组数

1. 堆顶元素，就是当前所有组中拥有最小右边界的组，因此新的元素会不会和已有组有相交，首先去看堆顶元素
   1. 如果此时遍历到的interval的start大于堆顶元素，即`queue.peek() < interval[0]`，该组就应该被动态更新了，其右边界应该变为`interval[0]`，并且堆顶元素出队
   2. 如果此时遍历到的interval的start小于堆顶元素，那么就说明需要新开一个组来存放该元素了，直接入队
2. 因为用的最小堆，所以如果interval的start小于堆顶元素，那么它一定小于之后的所有元素



