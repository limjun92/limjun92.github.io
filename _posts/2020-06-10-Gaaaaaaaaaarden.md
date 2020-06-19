---
title:  "Gaaaaaaaaaarden"
excerpt: "BFS 시뮬레이션 백준 Java"

categories:
  - Baekjoon_Algorithm
tags:
  - BFS
  - 백준
  - Java_algorithm
  - 시뮬레이션
last_modified_at: 2020-06-10
---
* 풀이시간 약 3시간 (combination을 두번 사용하는 부분, bfs를 한번만 사용하여 구현하는 부분)
* bfs를 구현하면서 Green을 먼저 다 뿌진후에 Red를 뿌리기 때문에 Geean에서 Red로 변하는 순간에 집중해서 문제를 풀이했다.
* 동시에 겹쳐지는 순간을 찾기 위해 3차 배열을 사용해서 구현했다.  


```java
  import java.util.ArrayList;
  import java.util.LinkedList;
  import java.util.Queue;
  import java.util.Scanner;

  public class Gaaaaaaaaaarden {

    public static void main(String[] args) {

      Scanner sc = new Scanner(System.in);

      N = sc.nextInt();
      M = sc.nextInt();

      arr = new int[N][M];

      G = sc.nextInt();
      R = sc.nextInt();

      ArrayList<Node> list = new ArrayList<>();

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
          arr[i][j] = sc.nextInt();
          if (arr[i][j] == 2)
            list.add(new Node(i, j));
        }
      }

      c(list, 0, 0, new Node[G]);

      System.out.println(max);

    }

    static int[][] arr;

    static int M;
    static int N;
    static int G;
    static int R;

    static int max = 0;

    static void c(ArrayList<Node> list, int n, int c, Node[] re) {
      if (re.length == c) {
        ArrayList<Node> tmplist = new ArrayList<>();
        int cnt = 0;

        for (int i = 0; i < list.size(); i++) {
          if (cnt < re.length && list.get(i).r == re[cnt].r && list.get(i).c == re[cnt].c) {
            cnt++;
          } else {
            tmplist.add(list.get(i));
          }
        }
        c2(tmplist, 0, 0, new Node[R], re);
        return;
      }
      if (list.size() == n)
        return;
      re[c] = list.get(n);
      c(list, n + 1, c + 1, re);
      c(list, n + 1, c, re);

    }

    static void c2(ArrayList<Node> tmplist, int n, int c, Node[] re2, Node[] re) {
      if (re2.length == c) {
        Queue<re_Node> q = new LinkedList<>();
        int[][][] tmp = new int[2][N][M];
        for (int i = 0; i < re.length; i++) {
          q.add(new re_Node(re[i].r, re[i].c, 2, 0));
          tmp[0][re[i].r][re[i].c] = 1;
        }
        for (int i = 0; i < re2.length; i++) {
          q.add(new re_Node(re2[i].r, re2[i].c, 2, 1));
          tmp[1][re2[i].r][re2[i].c] = 1;
        }
        int cnt = 0;

        while (!q.isEmpty()) {
          re_Node node = q.poll();
          if (tmp[0][node.r][node.c] == -1 || tmp[1][node.r][node.c] == -1)
            continue;
          for (int i = 0; i < 4; i++) {
            int nr = node.r + dr[i];
            int nc = node.c + dc[i];
            if (nr < 0 || nr >= N || nc < 0 || nc >= M || arr[nr][nc] == 0 || tmp[node.type][nr][nc] != 0)
              continue;
            if (node.type == 0 && tmp[1][nr][nc] != 0)
              continue;
            if (node.type == 1 && tmp[0][nr][nc] == node.cnt) {
              cnt++;
              tmp[0][nr][nc] = -1;
              tmp[1][nr][nc] = -1;
              continue;
            } else if (node.type == 1 && tmp[0][nr][nc] != 0)
              continue;
            tmp[node.type][nr][nc] = node.cnt;
            q.add(new re_Node(nr, nc, node.cnt + 1, node.type));
          }
        }
        max = Math.max(max, cnt);
        return;
      }
      if (tmplist.size() == n)
        return;
      re2[c] = tmplist.get(n);
      c2(tmplist, n + 1, c + 1, re2, re);
      c2(tmplist, n + 1, c, re2, re);
    }

    static int[] dr = { -1, 1, 0, 0 };
    static int[] dc = { 0, 0, -1, 1 };

    static class re_Node {
      int r, c, cnt, type;

      re_Node(int r, int c, int cnt, int type) {
        this.r = r;
        this.c = c;
        this.cnt = cnt;
        this.type = type;
      }
    }
    static class Node {
      int r, c;

      Node(int r, int c) {
        this.r = r;
        this.c = c;
      }
    }

  }
```
