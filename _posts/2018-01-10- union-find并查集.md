---
layout:     post
title:      union-find并查集
subtitle:   
date:       2017-02-15
author:     BY
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 算法
---
# 思想：
## quickfind
  使用数组a记录每个元素编号，初始化为节点编号，find的过程：查看数组元素是否相同，union过程将所有a[q]的编号改为a[p]的编号，代价为N
  
```
package union_find;

/**
 * Created by zxl on 2017/11/17.
 */
public class QucikFind {

    private int[] id;
    //构造函数
    public QucikFind(int N){
        id = new int[N];
        for(int i=0;i<N;i++){
            id[i]=i;
        }
    }

    //判断是否连接
    public boolean connected(int q,int p){
        return id[q]==id[p];
    }

    //合并
    public void union(int p,int q){
        int pid = id[p];
        int qid = id[q];
        for(int i = 0;i<id.length;i++){
            if(id[i] == pid){
                id[i]=qid;
            }
        }
    }

}
```

## quickunion
  使用数组a记录每个元素编号，初始化为节点编号，find的过程：寻找两个元素的根节点是否相同，union过程：将p的根节点的父节点之前q的根节点 ，find/union代价为树的高度
  
```
public class QucikUnion {
    private int[] id;

    //构造函数
    public QucikUnion(int N){
        id = new int[N];
        for(int i=0;i<N;i++){
            id[i]=i;
        }
    }

    public int root(int i){
        while(i != id[i]){i=id[i];}
        return i;
    }

    public boolean find(int p,int q){
        int proot = root(p);
        int qroot = root(q);
        return proot==qroot;
    }

    public void union(int p,int q){
        int proot = root(p);
        int qroot = root(q);
        if(proot==qroot) return;
        id[proot]=qroot;
    }
 }
```

## weighted union find
使用数组a记录每个元素编号，初始化为节点编号，使用数组size记录每个联通域或者树的大小，find过程：寻找两个元素的根节点是否相同（root函数），union过程：若size小的树的根节点指向size大的树的根节点，find/union代价为logN

```
package union_find;

/**
 * Created by zxl on 2017/11/17.
 */
public class WeightedQuickUnion {
    private int[] id;
    private int[] sz;

    public  WeightedQuickUnion(int N){
        id = new int[N];
        sz = new int[N];
        for(int i = 0;i<N;i++){
            id[i]=i;
            sz[i]=1;
        }
    }

    public int root(int i){
        while(i != id[i]){
            //路径压缩
            id[i] = id[id[i]];
            i=id[i];
        }
        return i;
    }

    public boolean connected(int p,int q){
        return root(p) ==root(q);
    }

    public void union(int p,int q){
        int proot=root(p);
        int qroot=root(q);
        if(proot==qroot) return;
        if(sz[proot]<sz[qroot]){
            id[proot] = qroot;
            sz[qroot] +=sz[proot];
        }else {
            id[qroot] = proot;
            sz[proot] = qroot;
        }

    }
}
```
