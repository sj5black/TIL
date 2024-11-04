# 너비 우선 탐색 (BFS, Breadth-First Search)

- 시작 노드에서 가까운 노드부터 차례대로 탐색하는 방식
- 한 레벨의 모든 노드를 먼저 탐색한 후, 그다음 레벨의 노드를 탐색
- 큐(Queue) 자료구조를 사용하여 구현
- 먼저 방문한 노드를 큐에 넣고, 큐에서 차례대로 꺼내면서 해당 노드와 연결된 노드를 다시 큐에 넣는다

### BFS 동작 예시

1. 시작 노드를 큐에 넣음.
2. 큐에서 노드를 하나씩 꺼내어 방문.
3. 그 노드와 인접한 모든 노드를 큐에 넣음.
4. 큐가 빌 때까지 2~3 과정을 반복.

### BFS 특징
- 모든 노드를 층별로 탐색.
- 최단 경로 탐색에 적합.
- 시간이 오래 걸릴 수 있으므로 노드가 많은 그래프에서 비효율적

```python
from collections import deque

def bfs(graph, start):
    visited = []
    queue = deque([start])

    while queue :
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])
    
    return visited

graph = {
    'A' : ['B', 'C'],
    'B' : ['A', 'D', 'E'],
    'C' : ['A', 'F'],
    'D' : ['B'],
    'E' : ['B', 'F'],
    'F' : ['C', 'E'],
}

print(bfs(graph, 'A'))
# Output: ['A', 'B', 'C', 'D', 'E', 'F']
```
---
# 깊이 우선 탐색 (DFS, Depth-First Search)
- 그래프의 한 경로를 끝까지 탐색하고, 막다른 길에 도달하면 뒤로 돌아와 다른 경로를 탐색하는 방식
- 스택(Stack) 자료구조나 재귀 호출을 사용하여 구현
- 한 경로를 끝까지 탐색한 후, 다른 경로를 탐색

### DFS 동작 예시

1. 시작 노드부터 방문하며, 방문할 수 있는 깊이까지 탐색.
2. 더 이상 갈 곳이 없으면 이전 노드로 돌아옴.
3. 모든 경로를 탐색할 때까지 반복.

### DFS 특징
- 모든 노드를 한 경로를 깊게 탐색.
- 재귀적으로 구현 가능.
- 경로가 긴 경우, 메모리 사용량이 많아질 수 있음.
- 전체 경로를 탐색해야 할 때 적합.

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    
    visited.append(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited

graph = {
    'A' : ['B', 'C'],
    'B' : ['A', 'D', 'E'],
    'C' : ['A', 'F'],
    'D' : ['B'],
    'E' : ['B', 'F'],
    'F' : ['C', 'E'],
}

print (dfs(graph, 'A'))
# Output: ['A', 'B', 'D', 'E', 'F', 'C']
```
---
### 백준 1260번 : 그래프 탐색

**문제**
그래프를 DFS로 탐색한 결과와 BFS로 탐색한 결과를 출력하는 프로그램을 작성하시오. 단, 방문할 수 있는 정점이 여러 개인 경우에는 정점 번호가 작은 것을 먼저 방문하고, 더 이상 방문할 수 있는 점이 없는 경우 종료한다. 정점 번호는 1번부터 N번까지이다.

**입력**
첫째 줄에 정점의 개수 N(1 ≤ N ≤ 1,000), 간선의 개수 M(1 ≤ M ≤ 10,000), 탐색을 시작할 정점의 번호 V가 주어진다. 다음 M개의 줄에는 간선이 연결하는 두 정점의 번호가 주어진다. 어떤 두 정점 사이에 여러 개의 간선이 있을 수 있다. 입력으로 주어지는 간선은 양방향이다.

**출력**
첫째 줄에 DFS를 수행한 결과를, 그 다음 줄에는 BFS를 수행한 결과를 출력한다. V부터 방문된 점을 순서대로 출력하면 된다.


**풀이)**
1. 주어진 노드 개수를 토대로 각 노드에 인덱싱 부여 (1~N)
2. 간선 정보를 토대로 각 노드별 인접한 노드 리스트 생성 (양방향)
3. 리스트 sort (오름차순) 후 dictionary 형태로 저장
4. DFS, BFS 함수 생성 후 각 결과값   출력

**TMI**
```py
# 전체 컬럼의 모든 요소값을 기준으로 중복 검사
df.duplicated()

# 'name' 컬럼만을 기준으로 중복 검사
df.duplicated(subset=['name'])

# df의 각 컬럼들의 결측치 개수 확인
print(df.isnull().sum(axis=0))
```