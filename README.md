<p align="center"><img src="/images/InterviewsRepository.jpg?raw=true"></p>

# Interviews
> Your personal guide to Software Engineering technical interviews. Video
> solutions to the following interview problems with detailed explanations can be found [here](https://www.youtube.com/channel/UCKvwPt6BifPP54yzH99ff1g).
<a href="https://www.youtube.com/channel/UCKvwPt6BifPP54yzH99ff1g" style="display:block;"><img src="/images/youtube.jpg?raw=true"></a>
>
> Maintainer - [Kevin Naughton Jr../README-zh-cn.md)

## 목차
- [온라인 저지](#online-judges)
- [자료 구조](#data-structures)
- [알고리즘](#algorithms)
- [그리디 알고리즘](#greedy-algorithms)
- [비트마스크](#bitmasks)
- [런타임 분석](#runtime-analysis)
- [비디오 강의](#video-lectures)
- [디렉토리 구조](#directory-tree)

## 온라인 저지
* [백준 온라인저지](https://www.acmicpc.net)
* [Codility](https://codility.com/programmers/lessons/1-iterations/)
* [Code Forces](http://codeforces.com/)
* [Code Chef](https://www.codechef.com/)
* [Sphere Online Judge - SPOJ](http://www.spoj.com/)

## 자료구조
### 연결리스트(Linked List)
 * *연결리스트*는 노드라는 데이터 원소들로 이루어진 선형 컬렉션이며 각각의 노드는 포인터를 사용해서 다음 노드를 가리키고 있습니다. 노드들이 그룹을 이루는 자료구조이며 일련의 순서들을 표현합니다.
 * **단방향 연결리스트**: 각각의 노드가 다음 노드를 가리키며 마지막 노드는 `null`을 가리키는 연결리스트
 * **양방향 연결리스트**: 각각의 노드는 2개의 포인터 `p`와 `n`을 가지며 `p`는 이전노드를 가리키고 `n`은 다음 노드를 가리킵니다. 마지막 노드의 포인터 `n`은 `null`을 가리킵니다.
 * **순환 연결리스트**: 각각의 노드가 다음 노드를 가리키며 마지막 노드는 첫번째 노드를 가리킵니다.
 * **시간복잡도**:
   * 접근: `O(n)`
   * 탐색: `O(n)`
   * 삽입: `O(1)`
   * 삭제: `O(1)`
* [연결리스트 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%EB%A7%81%ED%81%AC%EB%93%9C%20%EB%A6%AC%EC%8A%A4%ED%8A%B8)
* [생활코딩 공동공부: 연결리스트](https://opentutorials.org/module/1335/8821)

### 스택(Stack)
 * *스택*은 원소들의 컬렉션이며, 원소를 추가하는 *push* 기능과 가장 최근 원소를 제거하는 *pop* 기능을 가집니다.
 * 가장 나중에 들어간 원소가 가장 먼저 나오기 때문에 **LIFO(Last In, First Out) 자료구조**입니다.
 * **시간복잡도**:
   * 접근: `O(n)`
   * 탐색: `O(n)`
   * 삽입: `O(1)`
   * 삭제: `O(1)`
* [스택 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%EC%8A%A4%ED%83%9D)

### 큐(Queue)
 * *큐*는 원소들의 컬렉션이며, 원소를 추가하는 *enqueue* 기능과 가장 오래된 원소를 제거하는 *dequeue* 기능을 가집니다.
 * 가장 처음에 들어간 원소가 가장 먼저 나오기 때문에 **FIFIO(First In, First Out) 자료구조**입니다.
 * **시간복잡도**:
   * 접근: `O(n)`
   * 탐색: `O(n)`
   * 삽입: `O(1)`
   * 삭제: `O(1)`
* [큐 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%ED%81%90)

### 트리(Tree)
 * *트리*는 방향이 없고, 연결되어있는, 사이클이 없는 그래프입니다.

### 이진 트리(Binary Tree)
 * *이진 트리*는 각 노드가 *left child*와 *right child*라고 불리는 자식 노드들을 최대 2개 가질 수 있는 트리 자료구조입니다.
 * **정 이진 트리(Full Tree)**: 모든 노드가 0개 혹은 2개의 자식노드를 가지는 이진 트리입니다.
 * **포화 이진 트리(Perfect Binary Tree)**: 안쪽 노드들이 전부 2개의 자식노드를 가지며 모든 리프 노드들이 같은 깊이를 가지는 이진 트리입니다.
 * **완전 이진 트리(Complete Binary Tree)**: 왼쪽 자식노드부터 채워지며 마지막 레벨을 제외하고는 모든 자식노드들이 채워져 있는 이진 트리입니다.

### 이진 탐색 트리(Binary Search Tree)
 * BST라고 불리는 이진 탐색 트리는 각 노드가 왼쪽 부분 트리나 오른쪽 부분 트리에 있는 어떤 값보다 크거나 같아야 한다는 속성을 가지는 이진 트리입니다.
 * **시간 복잡도**:
   * 접근: `O(log(n))`
   * 탐색: `O(log(n))`
   * 삽입: `O(log(n))`
   * 삭제: `O(log(n))`

<img src="https://github.com/kdn251/interviews/raw/master/images/BST.png?raw=true" width="300px">

* [이진 탐색 트리 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%EC%9D%B4%EC%A7%84%20%ED%83%90%EC%83%89%20%ED%8A%B8%EB%A6%AC)

### 트라이(Trie)
* 트라이는 기수(radix) 혹은 접두사(prefix) 트리라고도 불리며, 키가 문자열인 연관배열이나 동적인 집합을 저장하는데 사용되는 탐색 트리입니다. 어떤 노드도 그 노드에 연관된 키를 저장하지는 않습니다. 대신, 트리에서의 위치가 해당 노드와 연관된 키를 정의합니다. 노드의 모든 자식들은 그 노드에 연관된 문자열을 접두사로 가지며 루트노드는 빈 문자열을 가집니다.

<img src="https://github.com/kdn251/interviews/raw/master/images/trie.png?raw=true">

* [트라이 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%ED%8A%B8%EB%9D%BC%EC%9D%B4)

### 펜윅 트리(Fenwick Tree)
* 펜윅 트리는 이진 인덱스 트리라고도 불리는, 개념적으로는 트리이지만 실제로는 배열을 통한 암시적인 자료 구조로 구현됩니다. 배열에서 정점을 나타내는 인덱스가 주어지면 인덱스를 이진수로 표현한 뒤, 비트 연산을 통해서 정점의 부모나 자식 인덱스가 계산됩니다. 배열의 각 원소들은 미리 계산된 범위의 합을 가지고 있으며 루트노드로 이동하는 중에 만나는 추가적인 범위들을 결합하면 접두사의 합이 계산됩니다.
* **시간 복잡도**:
  * 부분합 계산: `O(log(n))`
  * 갱신: `O(log(n))`

<img src="https://github.com/kdn251/interviews/raw/master/images/fenwickTree.png?raw=true">

* [펜윅 트리에 관한 백준님 설명](https://www.acmicpc.net/blog/view/21)
* [펜윅 트리 문제를 풀어보자](https://www.acmicpc.net/problem/tag/%ED%8E%9C%EC%9C%85%20%ED%8A%B8%EB%A6%AC)

### 세그먼트 트리(Segment Tree)
* A Segment tree, is a tree data structure for storing intervals, or segments. It allows querying which of the stored segments contain
  a given point
* Time Complexity:
  * Range Query: `O(log(n))`
  * Update: `O(log(n))`

![Alt text](/images/segmentTree.png?raw=true "Segment Tree")

### Heap
* A *Heap* is a specialized tree based structure data structure that satisfies the *heap* property: if A is a parent node of
B, then the key (the value) of node A is ordered with respect to the key of node B with the same ordering applying across the entire heap.
A heap can be classified further as either a "max heap" or a "min heap". In a max heap, the keys of parent nodes are always greater
than or equal to those of the children and the highest key is in the root node. In a min heap, the keys of parent nodes are less than
or equal to those of the children and the lowest key is in the root node
* Time Complexity:
  * Access Max / Min: `O(1)`
  * Insert: `O(log(n))`
  * Remove Max / Min: `O(log(n))`

<img src="/images/heap.png?raw=true" alt="Max Heap" width="400" height="500">


### Hashing
* *Hashing* is used to map data of an arbitrary size to data of a fixed size. The values returned by a hash
  function are called hash values, hash codes, or simply hashes. If two keys map to the same value, a collision occurs
* **Hash Map**: a *hash map* is a structure that can map keys to values. A hash map uses a hash function to compute
  an index into an array of buckets or slots, from which the desired value can be found.
* Collision Resolution
 * **Separate Chaining**: in *separate chaining*, each bucket is independent, and contains a list of entries for each index. The
 time for hash map operations is the time to find the bucket (constant time), plus the time to iterate through the list
 * **Open Addressing**: in *open addressing*, when a new entry is inserted, the buckets are examined, starting with the
 hashed-to-slot and proceeding in some sequence, until an unoccupied slot is found. The name open addressing refers to
 the fact that the location of an item is not always determined by its hash value


![Alt text](/images/hash.png?raw=true "Hashing")

### Graph
* A *Graph* is an ordered pair of G = (V, E) comprising a set V of vertices or nodes together with a set E of edges or arcs,
  which are 2-element subsets of V (i.e. an edge is associated with two vertices, and that association takes the form of the
  unordered pair comprising those two vertices)
 * **Undirected Graph**: a graph in which the adjacency relation is symmetric. So if there exists an edge from node u to node
 v (u -> v), then it is also the case that there exists an edge from node v to node u (v -> u)
 * **Directed Graph**: a graph in which the adjacency relation is not symmetric. So if there exists an edge from node u to node v
 (u -> v), this does *not* imply that there exists an edge from node v to node u (v -> u)


<img src="/images/graph.png?raw=true" alt="Graph" width="400" height="500">

## Algorithms

### Sorting

#### Quicksort
* Stable: `No`
* Time Complexity:
  * Best Case: `O(nlog(n))`
  * Worst Case: `O(n^2)`
  * Average Case: `O(nlog(n))`

![Alt text](/images/quicksort.gif?raw=true "Quicksort")

#### Mergesort
* *Mergesort* is also a divide and conquer algorithm. It continuously divides an array into two halves, recurses on both the
  left subarray and right subarray and then merges the two sorted halves
* Stable: `Yes`
* Time Complexity:
  * Best Case: `O(nlog(n))`
  * Worst Case: `O(nlog(n))`
  * Average Case: `O(nlog(n))`

![Alt text](/images/mergesort.gif?raw=true "Mergesort")

#### Bucket Sort
* *Bucket Sort* is a sorting algorithm that works by distributing the elements of an array into a number of buckets. Each bucket
  is then sorted individually, either using a different sorting algorithm, or by recursively applying the bucket sorting algorithm
* Time Complexity:
  * Best Case: `Ω(n + k)`
  * Worst Case: `O(n^2)`
  * Average Case:`Θ(n + k)`

![Alt text](/images/bucketsort.png?raw=true "Bucket Sort")

#### Radix Sort
* *Radix Sort* is a sorting algorithm that like bucket sort, distributes elements of an array into a number of buckets. However, radix
  sort differs from bucket sort by 're-bucketing' the array after the initial pass as opposed to sorting each bucket and merging
* Time Complexity:
  * Best Case: `Ω(nk)`
  * Worst Case: `O(nk)`
  * Average Case: `Θ(nk)`

### Graph Algorithms

#### Depth First Search
* *Depth First Search* is a graph traversal algorithm which explores as far as possible along each branch before backtracking
* Time Complexity: `O(|V| + |E|)`

![Alt text](/images/dfsbfs.gif?raw=true "DFS / BFS Traversal")

#### Breadth First Search
* *Breadth First Search* is a graph traversal algorithm which explores the neighbor nodes first, before moving to the next
  level neighbors
* Time Complexity: `O(|V| + |E|)`

![Alt text](/images/dfsbfs.gif?raw=true "DFS / BFS Traversal")

#### Topological Sort
* *Topological Sort* is the linear ordering of a directed graph's nodes such that for every edge from node u to node v, u
  comes before v in the ordering
* Time Complexity: `O(|V| + |E|)`

#### Dijkstra's Algorithm
* *Dijkstra's Algorithm* is an algorithm for finding the shortest path between nodes in a graph
* Time Complexity: `O(|V|^2)`

![Alt text](/images/dijkstra.gif?raw=true "Dijkstra's")

#### Bellman-Ford Algorithm
* *Bellman-Ford Algorithm* is an algorithm that computes the shortest paths from a single source node to all other nodes in a weighted graph
* Although it is slower than Dijkstra's, it is more versatile, as it is capable of handling graphs in which some of the edge weights are
  negative numbers
* Time Complexity:
  * Best Case: `O(|E|)`
  * Worst Case: `O(|V||E|)`

![Alt text](/images/bellman-ford.gif?raw=true "Bellman-Ford")

#### Floyd-Warshall Algorithm
* *Floyd-Warshall Algorithm* is an algorithm for finding the shortest paths in a weighted graph with positive or negative edge weights, but
  no negative cycles
* A single execution of the algorithm will find the lengths (summed weights) of the shortest paths between *all* pairs of nodes
* Time Complexity:
  * Best Case: `O(|V|^3)`
  * Worst Case: `O(|V|^3)`
  * Average Case: `O(|V|^3)`

#### Prim's Algorithm
* *Prim's Algorithm* is a greedy algorithm that finds a minimum spanning tree for a weighted undirected graph. In other words, Prim's find a
  subset of edges that forms a tree that includes every node in the graph
* Time Complexity: `O(|V|^2)`

![Alt text](/images/prim.gif?raw=true "Prim's Algorithm")

#### Kruskal's Algorithm
* *Kruskal's Algorithm* is also a greedy algorithm that finds a minimum spanning tree in a graph. However, in Kruskal's, the graph does not
  have to be connected
* Time Complexity: `O(|E|log|V|)`

![Alt text](/images/kruskal.gif?raw=true "Kruskal's Algorithm")

## Greedy Algorithms
* *Greedy Algorithms* are algorithms that make locally optimal choices at each step in the hope of eventually reaching the globally optimal solution
* Problems must exhibit two properties in order to implement a Greedy solution:
 * Optimal Substructure
    * An optimal solution to the problem contains optimal solutions to the given problem's subproblems
 * The Greedy Property
    * An optimal solution is reached by "greedily" choosing the locally optimal choice without ever reconsidering previous choices
* Example - Coin Change
    * Given a target amount V cents and a list of denominations of n coins, i.e. we have coinValue[i] (in cents) for coin types i from [0...n - 1],
      what is the minimum number of coins that we must use to represent amount V? Assume that we have an unlimited supply of coins of any type
    * Coins - Penny (1 cent), Nickel (5 cents), Dime (10 cents), Quarter (25 cents)
    * Assume V = 41. We can use the Greedy algorithm of continuously selecting the largest coin denomination less than or equal to V, subtract that
      coin's value from V, and repeat.
    * V = 41 | 0 coins used
    * V = 16 | 1 coin used (41 - 25 = 16)
    * V = 6  | 2 coins used (16 - 10 = 6)
    * V = 1  | 3 coins used (6 - 5 = 1)
    * V = 0  | 4 coins used (1 - 1 = 0)
    * Using this algorithm, we arrive at a total of 4 coins which is optimal

## Bitmasks
* Bitmasking is a technique used to perform operations at the bit level. Leveraging bitmasks often leads to faster runtime complexity and
  helps limit memory usage
* Test kth bit: `s & (1 << k);`
* Set kth bit: `s |= (1 << k);`
* Turn off kth bit: `s &= ~(1 << k);`
* Toggle kth bit: `s ^= (1 << k);`
* Multiple by 2<sup>n</sup>: `s << n;`
* Divide by 2<sup>n</sup>: `s >> n;`
* Intersection: `s & t;`
* Union: `s | t;`
* Set Subtraction: `s & ~t;`
* Extract lowest set bit: `s & (-s);`
* Extract lowest unset bit: `~s & (s + 1);`
* Swap Values:
     ​    ​    ```
     ​    ​       x ^= y;
     ​    ​       y ^= x;
     ​    ​       x ^= y;
     ​    ​    ```

## Runtime Analysis

#### Big O Notation
* *Big O Notation* is used to describe the upper bound of a particular algorithm. Big O is used to describe worst case scenarios

![Alt text](/images/bigO.png?raw=true "Theta Notation")

#### Little O Notation
* *Little O Notation* is also used to describe an upper bound of a particular algorithm; however, Little O provides a bound
  that is not asymptotically tight

#### Big Ω Omega Notation
* *Big Omega Notation* is used to provide an asymptotic lower bound on a particular algorithm

![Alt text](/images/bigOmega.png?raw=true "Theta Notation")

#### Little ω Omega Notation
* *Little Omega Notation* is used to provide a lower bound on a particular algorithm that is not asymptotically tight

#### Theta Θ Notation
* *Theta Notation* is used to provide a bound on a particular algorithm such that it can be "sandwiched" between
  two constants (one for an upper limit and one for a lower limit) for sufficiently large values

![Alt text](/images/theta.png?raw=true "Theta Notation")

## Video Lectures
* Data Structures
    * [UC Berkeley Data Structures](https://archive.org/details/ucberkeley-webcast?&and[]=subject%3A%22Computer%20Science%22&and[]=subject%3A%22CS%22)
    * [MIT Advanced Data Structures](https://www.youtube.com/watch?v=T0yzrZL1py0&list=PLUl4u3cNGP61hsJNdULdudlRL493b-XZf&index=1)
* Algorithms
    * [MIT Introduction to Algorithms](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=1)
    * [MIT Advanced Algorithms](https://www.youtube.com/playlist?list=PL6ogFv-ieghdoGKGg2Bik3Gl1glBTEu8c)
    * [UC Berkeley Algorithms](https://archive.org/details/ucberkeley-webcast?&and[]=subject%3A%22Computer%20Science%22&and[]=subject%3A%22CS%22)

## Directory Tree

```
.
├── Array
│   ├── bestTimeToBuyAndSellStock.java
│   ├── findTheCelebrity.java
│   ├── gameOfLife.java
│   ├── increasingTripletSubsequence.java
│   ├── insertInterval.java
│   ├── longestConsecutiveSequence.java
│   ├── maximumProductSubarray.java
│   ├── maximumSubarray.java
│   ├── mergeIntervals.java
│   ├── missingRanges.java
│   ├── productOfArrayExceptSelf.java
│   ├── rotateImage.java
│   ├── searchInRotatedSortedArray.java
│   ├── spiralMatrixII.java
│   ├── subsetsII.java
│   ├── subsets.java
│   ├── summaryRanges.java
│   ├── wiggleSort.java
│   └── wordSearch.java
├── Backtracking
│   ├── androidUnlockPatterns.java
│   ├── generalizedAbbreviation.java
│   └── letterCombinationsOfAPhoneNumber.java
├── BinarySearch
│   ├── closestBinarySearchTreeValue.java
│   ├── firstBadVersion.java
│   ├── guessNumberHigherOrLower.java
│   ├── pow(x,n).java
│   └── sqrt(x).java
├── BitManipulation
│   ├── binaryWatch.java
│   ├── countingBits.java
│   ├── hammingDistance.java
│   ├── maximumProductOfWordLengths.java
│   ├── numberOf1Bits.java
│   ├── sumOfTwoIntegers.java
│   └── utf-8Validation.java
├── BreadthFirstSearch
│   ├── binaryTreeLevelOrderTraversal.java
│   ├── cloneGraph.java
│   ├── pacificAtlanticWaterFlow.java
│   ├── removeInvalidParentheses.java
│   ├── shortestDistanceFromAllBuildings.java
│   ├── symmetricTree.java
│   └── wallsAndGates.java
├── DepthFirstSearch
│   ├── balancedBinaryTree.java
│   ├── battleshipsInABoard.java
│   ├── convertSortedArrayToBinarySearchTree.java
│   ├── maximumDepthOfABinaryTree.java
│   ├── numberOfIslands.java
│   ├── populatingNextRightPointersInEachNode.java
│   └── sameTree.java
├── Design
│   └── zigzagIterator.java
├── DivideAndConquer
│   ├── expressionAddOperators.java
│   └── kthLargestElementInAnArray.java
├── DynamicProgramming
│   ├── bombEnemy.java
│   ├── climbingStairs.java
│   ├── combinationSumIV.java
│   ├── countingBits.java
│   ├── editDistance.java
│   ├── houseRobber.java
│   ├── paintFence.java
│   ├── paintHouseII.java
│   ├── regularExpressionMatching.java
│   ├── sentenceScreenFitting.java
│   ├── uniqueBinarySearchTrees.java
│   └── wordBreak.java
├── HashTable
│   ├── binaryTreeVerticalOrderTraversal.java
│   ├── findTheDifference.java
│   ├── groupAnagrams.java
│   ├── groupShiftedStrings.java
│   ├── islandPerimeter.java
│   ├── loggerRateLimiter.java
│   ├── maximumSizeSubarraySumEqualsK.java
│   ├── minimumWindowSubstring.java
│   ├── sparseMatrixMultiplication.java
│   ├── strobogrammaticNumber.java
│   ├── twoSum.java
│   └── uniqueWordAbbreviation.java
├── LinkedList
│   ├── addTwoNumbers.java
│   ├── deleteNodeInALinkedList.java
│   ├── mergeKSortedLists.java
│   ├── palindromeLinkedList.java
│   ├── plusOneLinkedList.java
│   ├── README.md
│   └── reverseLinkedList.java
├── Queue
│   └── movingAverageFromDataStream.java
├── README.md
├── Sort
│   ├── meetingRoomsII.java
│   └── meetingRooms.java
├── Stack
│   ├── binarySearchTreeIterator.java
│   ├── decodeString.java
│   ├── flattenNestedListIterator.java
│   └── trappingRainWater.java
├── String
│   ├── addBinary.java
│   ├── countAndSay.java
│   ├── decodeWays.java
│   ├── editDistance.java
│   ├── integerToEnglishWords.java
│   ├── longestPalindrome.java
│   ├── longestSubstringWithAtMostKDistinctCharacters.java
│   ├── minimumWindowSubstring.java
│   ├── multiplyString.java
│   ├── oneEditDistance.java
│   ├── palindromePermutation.java
│   ├── README.md
│   ├── reverseVowelsOfAString.java
│   ├── romanToInteger.java
│   ├── validPalindrome.java
│   └── validParentheses.java
├── Tree
│   ├── binaryTreeMaximumPathSum.java
│   ├── binaryTreePaths.java
│   ├── inorderSuccessorInBST.java
│   ├── invertBinaryTree.java
│   ├── lowestCommonAncestorOfABinaryTree.java
│   ├── sumOfLeftLeaves.java
│   └── validateBinarySearchTree.java
├── Trie
│   ├── addAndSearchWordDataStructureDesign.java
│   ├── implementTrie.java
│   └── wordSquares.java
└── TwoPointers
    ├── 3Sum.java
    ├── 3SumSmaller.java
    ├── mergeSortedArray.java
    ├── minimumSizeSubarraySum.java
    ├── moveZeros.java
    ├── removeDuplicatesFromSortedArray.java
    ├── reverseString.java
    └── sortColors.java

18 directories, 124 files
```
