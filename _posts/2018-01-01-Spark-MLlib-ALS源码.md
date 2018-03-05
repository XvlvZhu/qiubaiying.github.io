---
layout:     post
title:      Spark-MLlib-ALS源码
subtitle:   
date:       2018-01-01
author:     xvlvzhu
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Spark
---
## 1.Rating数据结构
A more compact class to represent a rating than Tuple3[Int, Int, Double].
```
case class Rating @Since(0.8.0) (
    @Since(0.8.0) user Int,
    @Since(0.8.0) product Int,
    @Since(0.8.0) rating Double)
```
---

## 2.ALS算法
### Alternating Least Squares matrix factorization
### 交替最小二乘矩阵分解

  
   ALS attempts to estimate the ratings matrix `R` as the product of **two lower-rank matrices(两个低秩矩阵)**,
   `X` and `Y`, i.e. `X  Yt = R`. Typically these approximations are called 'factor' matrices.
   The general approach is iterative. During each iteration, one of the factor matrices is held
   constant, while the other is solved for using least squares. The newly-solved factor matrix is
   then held constant while solving for the other factor matrix.
  
   This is a blocked implementation of the ALS factorization algorithm that groups the two sets
   of factors (referred to as users and products) into blocks and reduces communication by only
   sending one copy of each user vector to each product block on each iteration, and only for the
   product blocks that need that user's feature vector. This is achieved by precomputing some
   information about the ratings matrix to determine the out-links of each user (which blocks of
   products it will contribute to) and in-link information for each product (which of the feature
   vectors it receives from each user block it will depend on). This allows us to send only an
   array of feature vectors between each user block and product block, and have the product block
   find the users' ratings and update the products based on these messages.
  
   For implicit preference data, the algorithm used is based on
   Collaborative Filtering for Implicit Feedback Datasets, available at
   [[httpdx.doi.org10.1109ICDM.2008.22]], adapted for the blocked approach used here.
  
   Essentially instead of finding the low-rank approximations to the rating matrix `R`,
   this finds the approximations for a preference matrix `P` where the elements of `P` are 1 if
   r  0 and 0 if r = 0. The ratings then act as 'confidence' values related to strength of
   indicated user
   preferences rather than explicit ratings given to items.
  

## 3.参数：

```
private var numUserBlocks Int,   设置并行计算的user block
private var numProductBlocks Int,  设置并行计算的product block
private var rank Int,  设置特征矩阵的维数
private var iterations Int, 设置迭代次数
private var lambda Double, 正则化参数，默认0.01
private var implicitPrefs Boolean, 是否使用隐藏偏好
private var alpha Double, 设置用于计算ALS的置信度的常数
private var seed Long = System.nanoTime() 设置一个随机种子以确定性结果
```

## 4.run
     Run ALS with the configured parameters on an input RDD of (user, product, rating) triples.
     Returns a MatrixFactorizationModel with feature vectors for each user and product.

```
  def run(ratings RDD[Rating]) MatrixFactorizationModel = {
    val sc = ratings.context
    
    如果user block,product block设置为-1,则自动计算并行度
    val numUserBlocks = if (this.numUserBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.size  2)
    } else {
      this.numUserBlocks
    }
    val numProductBlocks = if (this.numProductBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.size  2)
    } else {
      this.numProductBlocks
    }
    
    跳转到org.apache.spark.ml.recommendation.NewALS算法训练模型
    val (floatUserFactors, floatProdFactors) = NewALS.train[Int](
      ratings = ratings.map(r = NewALS.Rating(r.user, r.product, r.rating.toFloat)),
      rank = rank,
      numUserBlocks = numUserBlocks,
      numItemBlocks = numProductBlocks,
      maxIter = iterations,
      regParam = lambda,
      implicitPrefs = implicitPrefs,
      alpha = alpha,
      nonnegative = nonnegative,
      intermediateRDDStorageLevel = intermediateRDDStorageLevel,
      finalRDDStorageLevel = StorageLevel.NONE,
      checkpointInterval = checkpointInterval,
      seed = seed)

    val userFactors = floatUserFactors
      .mapValues(_.map(_.toDouble))
      .setName(users)
      .persist(finalRDDStorageLevel)
    val prodFactors = floatProdFactors
      .mapValues(_.map(_.toDouble))
      .setName(products)
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userFactors.count()
      prodFactors.count()
    }
    new MatrixFactorizationModel(rank, userFactors, prodFactors)
  }
```

### 跳转到newALS的train方法

```
def train[ID ClassTag](  scalastyleignore
      ratings RDD[Rating[ID]],
      rank Int = 10,
      numUserBlocks Int = 10,
      numItemBlocks Int = 10,
      maxIter Int = 10,
      regParam Double = 1.0,
      implicitPrefs Boolean = false,
      alpha Double = 1.0,
      nonnegative Boolean = false,
      intermediateRDDStorageLevel StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval Int = 10,
      seed Long = 0L)(
      implicit ord Ordering[ID]) (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])])
```      

### 初始化ALSPartitioner和LocalIndexEncoder     
    
```
require(intermediateRDDStorageLevel != StorageLevel.NONE,
      ALS is not designed to run without persisting intermediate RDDs.)
    val sc = ratings.sparkContext
     基于hash的分区 
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
```

### 根据nonnegative参数选择解决矩阵分解的方法
 如果需要解的值为非负,即nonnegative为true，那么用非负最小二乘（NNLS）来解，如果没有这个限制，用乔里斯基（Cholesky）分解来解。
   
```
val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
```

### 将ratings数据转换为分区的格式 
 将ratings数据转换为分区的形式，即（（用户分区id，商品分区id），分区数据集blocks））的形式，并缓存到内存中。其中分区id的计算是通过ALSPartitioner的getPartitions方法获得的，分区数据集由RatingBlock组成，它表示（用户分区id，商品分区id ）对所对应的用户id集，商品id集，以及打分集，即（用户id集，商品id集，打分集）。
   
```
val blockRatings = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
```

    val (userInBlocks, userOutBlocks) =
      makeBlocks(user, blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
     materialize blockRatings and user blocks
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks) =
      makeBlocks(item, swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
     materialize item blocks
    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)
    var userFactors = initialize(userInBlocks, rank, seedGen.nextLong())
    var itemFactors = initialize(itemInBlocks, rank, seedGen.nextLong())
    var previousCheckpointFile Option[String] = None
    val shouldCheckpoint Int = Boolean = (iter) =
      sc.checkpointDir.isDefined && (iter % checkpointInterval == 0)
    val deletePreviousCheckpointFile () = Unit = () =
      previousCheckpointFile.foreach { file =
        try {
          FileSystem.get(sc.hadoopConfiguration).delete(new Path(file), true)
        } catch {
          case e IOException =
            logWarning(sCannot delete checkpoint file $file, e)
        }
      }
    if (implicitPrefs) {
      for (iter - 1 to maxIter) {
        userFactors.setName(suserFactors-$iter).persist(intermediateRDDStorageLevel)
        val previousItemFactors = itemFactors
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, implicitPrefs, alpha, solver)
        previousItemFactors.unpersist()
        itemFactors.setName(sitemFactors-$iter).persist(intermediateRDDStorageLevel)
         TODO Generalize PeriodicGraphCheckpointer and use it here.
        if (shouldCheckpoint(iter)) {
          itemFactors.checkpoint()  itemFactors gets materialized in computeFactors.
        }
        val previousUserFactors = userFactors
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, implicitPrefs, alpha, solver)
        if (shouldCheckpoint(iter)) {
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        previousUserFactors.unpersist()
      }
    } else {
      for (iter - 0 until maxIter) {
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, solver = solver)
        if (shouldCheckpoint(iter)) {
          itemFactors.checkpoint()
          itemFactors.count()  checkpoint item factors and cut lineage
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, solver = solver)
      }
    }
    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =
        items.flatMap { case (_, (ids, factors)) =
          ids.view.zip(factors)
        }
       Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
       and userFactors.
      }, preservesPartitioning = true)
      .setName(userFactors)
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =
        items.flatMap { case (_, (ids, factors)) =
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName(itemFactors)
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

## partionRating

```
  private def partitionRatings[ID ClassTag](
      ratings RDD[Rating[ID]],
      srcPart Partitioner,
      dstPart Partitioner) RDD[((Int, Int), RatingBlock[ID])] = {


    val numPartitions = srcPart.numPartitions  dstPart.numPartitions
    ratings.mapPartitions { iter =
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        val idx = srcBlockId + srcPart.numPartitions  dstBlockId
        val builder = builders(idx)
        builder.add(r)
        if (builder.size = 2048) {  2048  (3  4) = 24k
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size  0).map { case (block, idx) =
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx  srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName(ratingBlocks)
  }
```



## 初始化ALSPartitioner 和 LocalIndexEncoder
 LocalIndexEncoder对（blockid，localindex）即（分区id，分区内索引）进行编码，并将其转换为一个整数，这个整数在高位存分区ID，在低位存对应分区的索引，在空间上尽量做到了不浪费。同时也可以根据这个转换的整数分别获得blockid和localindex。这两个对象在后续的代码中会用到。

```
private[recommendation] class LocalIndexEncoder(numBlocks Int) extends Serializable {

    require(numBlocks  0, snumBlocks must be positive but found $numBlocks.)

    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1  numLocalIndexBits) - 1

     Encodes a (blockId, localIndex) into a single integer. 
    def encode(blockId Int, localIndex Int) Int = {
      require(blockId  numBlocks)
      require((localIndex & ~localIndexMask) == 0)
      (blockId  numLocalIndexBits)  localIndex
    }

     Gets the block id from an encoded index. 
    @inline
    def blockId(encoded Int) Int = {
      encoded  numLocalIndexBits
    }

     Gets the local index from an encoded index. 
    @inline
    def localIndex(encoded Int) Int = {
      encoded & localIndexMask
    }
  }
```


