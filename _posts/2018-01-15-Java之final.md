---
layout:     post
title:      Java之final
subtitle:   final关键字的一些用法
date:       2018-01-15
author:     xvlvzhu
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Java
---
# final变量
经常与static一同使用，表示只读变量
```
public static final String LOAN = "loan";
```
# final方法
表示这个方法不能被子类的方法重写，
==final方法比非final要快，因为在编译时已经静态绑定了，不需要动态绑定==
```
class PersonalLoan{
    public final String getName(){
        return "personal loan";
    }
}
 
class CheapPersonalLoan extends PersonalLoan{
    @Override
    public final String getName(){
        return "cheap personal loan"; //compilation error: overridden method is final
    }
}
```
# final类
final类功能通常是完整的，不能被继承，String，Integer及其他包装类等
```
final class PersonalLoan{
}
class CheapPersonalLoan extends PersonalLoan{  //compilation error: cannot inherit from final class
}
```

# final优点
1. final关键字提高了性能。JVM和Java应用都会缓存final变量。
2. final变量可以安全的在多线程环境下进行共享，而不需要额外的同步开销。
3. 使用final关键字，JVM会对方法、变量及类进行优化。

# final知识点
- final关键字可以用于成员变量、本地变量、方法以及类。
- final成员变量必须在声明的时候初始化或者在构造器中初始化，否则就会报编译错误。
- 你不能够对final变量再次赋值。
- 本地变量必须在声明时赋值。
- 在匿名类中所有变量都必须是final变量。
- final方法不能被重写。
- final类不能被继承。
- final关键字不同于finally关键字，后者用于异常处理。
- final关键字容易与finalize()方法搞混，后者是在Object类中定义的方法，是在垃圾回收之前被JVM调用的方法。
- 接口中声明的所有变量本身是final的。
- final和abstract这两个关键字是反相关的，final类就不可能是abstract的。
- final方法在编译阶段绑定，称为静态绑定(static binding)。
- 没有在声明时初始化final变量的称为空白final变量(blank final variable)，它们必须在构造器中初始化，或者调用this()初始化。不这么做的话，编译器会报错“final变量(变量名)需要进行初始化”。
- 将类、方法、变量声明为final能够提高性能，这样JVM就有机会进行估计，然后优化。
- 按照Java代码惯例，final变量就是常量，而且通常常量名要大写：

```
private final int COUNT = 10;
```

- 对于集合对象声明为final指的是引用不能被更改，但是你可以向其中增加，删除或者改变内容。譬如：

```
private final List Loans = new ArrayList();
list.add(“home loan”);  //valid
list.add("personal loan"); //valid
loans = new Vector();  //not valid
```
