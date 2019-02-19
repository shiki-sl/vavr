package com.itheima.lambda;

import io.vavr.*;
import io.vavr.Lazy;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Stream;
import io.vavr.control.Either;
import io.vavr.control.Option;
import io.vavr.control.Try;
import lombok.Data;
import org.junit.Test;

import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.function.Function;

import static io.vavr.API.*;

/**
 * @Author: shiki
 * @Date: 2019/1/26 19:36
 * 函数式编程vavr入门
 * <a>https://www.ibm.com/developerworks/cn/java/j-understanding-functional-programming-4/index.html?ca=drs-</a>
 */

public class FunctionText {

    private Function1<Integer, Integer> function1;
    private Function2<Integer, Integer, Integer> function2;
    private Function3<Integer, Integer, Integer, Integer> function3;
    private Function4<Integer, Integer, Integer, Integer, Integer> function4;

    /**
     * 组合
     */
    @Test
    public void group() {
        function3 = (v1, v2, v3) -> (v1 + v2) * v3;
        Function3<Integer, Integer, Integer, Integer> composed
                = function3.andThen(v -> v * 100);
        int result = composed.apply(1, 2, 3);
        System.out.println(result);

        Function1<String, String> function1 = String::toUpperCase;
        Function1<Object, String> toUpperCase = function1.compose(Object::toString);
        String str = toUpperCase.apply(List.of("aaa", "bbb", 123));
        System.out.println(str);
    }

    /**
     * function函数的部分应用
     */
    @Test
    public void part() {
        function4 = (v1, v2, v3, v4) -> (v1 + v2) * (v3 + v4);
        function2 = function4.apply(1, 2);
        int result = function2.apply(4, 5);
        System.out.println(result);
        Function1<Integer, Integer> function1 = function4.apply(1, 2, 3);
        System.out.println(function1.apply(-3));
    }

    /**
     * function函数的柯里化
     */
    @Test
    public void curried() {
        function3 = (v1, v2, v3) -> (v1 + v2) * v3;
        int result = function3.
                curried().apply(5).
                curried().apply(5).
                curried().apply(10);
        System.out.println(result);
    }

    /**
     * 记忆化方法
     * 使用记忆化的函数会根据参数值来缓存之前计算的结果。对于同样的参数值，再次的调用会返回缓存的值，而不需要再次计算。
     * 这是一种典型的以空间换时间的策略。可以使用记忆化的前提是函数有引用透明性
     * <p>
     * 原始的函数实现中使用 BigInteger 的 pow 方法来计算乘方。使用 memoized 方法可以得到该函数的记忆化版本。
     * 接着使用同样的参数调用两次并记录下时间。从结果可以看出来，第二次的函数调用的时间非常短，因为直接从缓存中获取结果。
     */
    @Test
    public void memory() {
        //            底        幂      值
        Function2<BigInteger, Integer, BigInteger> pow = BigInteger::pow;
        Function2<BigInteger, Integer, BigInteger> memoized = pow.memoized();
        long start = System.currentTimeMillis();
        memoized.apply(BigInteger.valueOf(1024), 1024);
        long end1 = System.currentTimeMillis();
        memoized.apply(BigInteger.valueOf(1024), 1024);
        long end2 = System.currentTimeMillis();
        System.out.println("%d ms " + (end1 - start) + "\r\n%d ms " + (end2 - end1));
    }

    /**
     * Vavr 中的 Option 与 Java 8 中的 Optional 是相似的。不过 Vavr 的 Option 是一个接口，
     * 有两个实现类 Option.Some 和 Option.None，分别对应有值和无值两种情况。
     * 使用 Option.some 方法可以创建包含给定值的 Some 对象，而 Option.none 可以获取到 None 对象的实例。
     * Option 也支持常用的 map、flatMap 和 filter 等操作
     */
    @Test
    public void option() {
        Option<String> str = Option.of("hello");
        str.map(String::length);
        System.out.println(str);
        str.flatMap(v -> Option.of(v.length()));
        System.out.println(str.get());
    }

    /**
     * Either
     * Either 表示可能有两种不同类型的值，分别称为左值或右值。只能是其中的一种情况。
     * Either 通常用来表示成功或失败两种情况。惯例是把成功的值作为右值，而失败的值作为左值。
     * 可以在 Either 上添加应用于左值或右值的计算。应用于右值的计算只有在 Either 包含右值时才生效，对左值也是同理。
     * 根据随机的布尔值来创建包含左值或右值的 Either 对象。Either 的 map 和 mapLeft 方法分别对右值和左值进行计算。
     */
    @Test
    public void either() {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        Either<String, String> either = eitherChild(random)
                .map(str -> str + "World")
                .mapLeft(Throwable::getMessage);
        System.out.println(either);
    }

    private Either<Throwable, String> eitherChild(ThreadLocalRandom random) {
        return random.nextBoolean()
                ? Either.left(new RuntimeException("Boom!"))
                : Either.right("Hello");
    }

    /**
     * Try
     * Try 用来表示一个可能产生异常的计算。Try 接口有两个实现类，Try.Success 和 Try.Failure，分别表示成功和失败的情况。
     * Try.Success 封装了计算成功时的返回值，而 Try.Failure 则封装了计算失败时的 Throwable 对象。
     * Try 的实例可以从接口 CheckedFunction0、Callable、Runnable 或 Supplier 中创建。Try 也提供了 map 和 filter 等方法。
     * 值得一提的是 Try 的 recover 方法，可以在出现错误时根据异常进行恢复。
     * <p>
     * 在清单 8 中，第一个 Try 表示的是 1/0 的结果，显然是异常结果。使用 recover 来返回 1。
     * 第二个 Try 表示的是读取文件的结果。由于文件不存在，Try 表示的也是异常。
     */
    @Test
    public void tryTest() {
        Try<Integer> result = Try.of(() -> 1 / 0).recover(e -> 1);
        System.out.println(result);
        Try<String> lines = Try.of(() -> Files.readAllLines(Paths.get("C:\\Users\\shiki\\Desktop\\BT种子\\预习顺序.txt")))
                .map(list -> String.join(",", list))
                .andThen((Consumer<String>) System.out::println);
        System.out.println(lines);
    }

    /**
     * Lazy
     * Lazy 表示的是一个延迟计算的值。在第一次访问时才会进行求值操作，而且该值只会计算一次。之后的访问操作获取的是缓存的值。
     * 在清单 9 中，Lazy.of 从接口 Supplier 中创建 Lazy 对象。方法 isEvaluated 可以判断 Lazy 对象是否已经被求值。
     */
    @Test
    public void lazy() {
        Lazy<BigInteger> lzay = Lazy.of(() ->
                BigInteger.valueOf(1024).pow(1024));
        System.out.println(lzay.isEvaluated());
        System.out.println(lzay.get());
        System.out.println(lzay.isEvaluated());
    }

    /**
     * 数据结构
     * Vavr 重新在 Iterable 的基础上实现了自己的集合框架。Vavr 的集合框架侧重在不可变上。Vavr 的集合类在使用上比 Java 流更简洁。
     * <p>
     * <p>
     * Vavr 的 Stream 提供了比 Java 中 Stream 更多的操作。可以使用 Stream.ofAll 从 Iterable 对象中创建出 Vavr 的 Stream。
     * 下面是一些 Vavr 中添加的实用操作：
     * <p>
     * groupBy：使用 Fuction 对元素进行分组。结果是一个 Map，Map 的键是分组的函数的结果，而值则是包含了同一组中全部元素的 Stream。
     * <p>
     * partition：使用 Predicate 对元素进行分组。结果是包含 2 个 Stream 的 Tuple2。Tuple2 的第一个 Stream 的元素满足
     * Predicate 所指定的条件，第二个 Stream 的元素不满足 Predicate 所指定的条件。
     * <p>
     * scanLeft 和 scanRight：分别按照从左到右或从右到左的顺序在元素上调用 Function，并累积结果。
     * <p>
     * zip：把 Stream 和一个 Iterable 对象合并起来，返回的结果 Stream 中包含 Tuple2 对象。Tuple2 对象的两个元素分别来自 Stream 和 Iterable 对象。
     * <p>
     * 在清单 10 中，第一个 groupBy 操作把 Stream 分成奇数和偶数两组；第二个 partition 操作把 Stream 分成大于 2 和不大于 2 两组；
     * 第三个 scanLeft 对包含字符串的 Stream 按照字符串长度进行累积；最后一个 zip 操作合并两个流，
     * 所得的结果 Stream 的元素数量与长度最小的输入流相同。
     * <p>
     * 清单 10. Stream 的使用示例
     */
    @Test
    public void strean() {
        Map<Boolean, List<Integer>> booleanListMap = Stream.ofAll(1, 2, 3, 4, 5)
                .groupBy(v -> v % 2 == 0)
                .mapValues(Value::toList);
        System.out.println(booleanListMap);

        Tuple2<List<Integer>, List<Integer>> listListTuple2 = Stream.ofAll(1, 2, 3, 4)
                .partition(v -> v > 2)
                .map(Value::toList, Value::toList);
        System.out.println(listListTuple2);

        List<Integer> integers = Stream.ofAll(List.of("helli", "world", "a"))
                .scanLeft(0, (sum, str) -> sum + str.length())
                .toList();
        System.out.println(integers);

        List<Tuple2<Integer, String>> tuple2List = Stream.ofAll(1, 2, 3)
                .zip(List.of("a", "b"))
                .toList();
        System.out.println(tuple2List.asJava());
    }

    /**
     * 模式匹配
     * <p>
     * 在 Java 中，我们可以使用 switch 和 case 来根据值的不同来执行不同的逻辑。
     * 不过 switch 和 case 提供的功能很弱，只能进行相等匹配。Vavr 提供了模式匹配的 API，
     * 可以对多种情况进行匹配和执行相应的逻辑。在清单 12 中，我们使用 Vavr 的 Match 和 Case 替换了 Java 中的
     * switch 和 case。Match 的参数是需要进行匹配的值。Case 的第一个参数是匹配的条件，用 Predicate 来表示；
     * 第二个参数是匹配满足时的值。$(value) 表示值为 value 的相等匹配，而 $() 表示的是默认匹配，相当于 switch 中的 default。
     */
    @Test
    public void method() {
        String input = "g";
        String result = Match(input).of(
                Case($("g"), "good"),
                Case($("b"), "bad"),
                Case($(), "unknown")
        );
        System.out.println(result);
    }

    /**
     * 在清单 13 中，我们用 $(v -> v > 0) 创建了一个值大于 0 的 Predicate。这里匹配的结果不是具体的值，而是通过 run 方法来产生副作用。
     */
    @Test
    public void runText() {
        int value = -1;
        Match(value).of(
                Case($(v -> v > 0), o -> run(() -> System.out.println("> 0"))),
                Case($(0), o -> run(() -> System.out.println("0"))),
                Case($(), o -> run(() -> System.out.println("< 0")))
        );
    }

    /**
     * 函数的副作用与组合方式
     */
    @Test
    public void monadTest() {
        class t {
            private Tuple2<Integer, Integer> increase1(int x) {
                return Tuple.of(x + 1, 1);
            }

            private Tuple2<Integer, Integer> decrease1(int x) {
                return Tuple.of(x - 1, 1);
            }

            private Function<Integer, Tuple2<Integer, Integer>> compose(
                    Function<Integer, Tuple2<Integer, Integer>> func1,
                    Function<Integer, Tuple2<Integer, Integer>> func2) {
                return x -> {
                    Tuple2<Integer, Integer> result1 = func1.apply(x);
                    Tuple2<Integer, Integer> result2 = func2.apply(result1._1);
                    return Tuple.of(result2._1, result1._2 + result2._2);
                };
            }

            private Tuple2<Integer, Integer> doCompose(int x) {
                return compose(this::increase1, this::decrease1).apply(x);
            }

        }
        t t = new t();
        System.out.println(t.doCompose(5));
    }

    /**
     * Writer Monad
     * <p>
     * 基于函数式编程副作用打印日志
     */
    @Test
    public void loggingMonad() {
        Function<Integer, LoggingMonad<Integer>> transform1 =
                v -> new LoggingMonad<>(v * 4, java.util.List.of(v + " * 4"));
        Function<Integer, LoggingMonad<Integer>> transform2 =
                v -> new LoggingMonad<>(v / 2, java.util.List.of(v + " / 4"));
        Function<Integer, LoggingMonad<Integer>> transform3 =
                v -> new LoggingMonad<>(v << 10, java.util.List.of(v + " >> 2"));
        Function<Integer, LoggingMonad<Integer>> transform4 =
                v -> new LoggingMonad<>(v >> 2, java.util.List.of(v + " >>> 2"));
        final LoggingMonad<Integer> result = LoggingMonad.pipeline(LoggingMonad.unit(16),
                java.util.List.of(transform1, transform2));
        final LoggingMonad<Integer> result1 = LoggingMonad.pipeline(LoggingMonad.unit(1),
                java.util.List.of(transform3, transform4));
        System.out.println(result);
        System.out.println(result1);
    }

    @Data
    private static class LoggingMonad<T> {
        private final T value;
        private final java.util.List<String> logs;

        static <T> LoggingMonad<T> unit(T value) {
            return new LoggingMonad<>(value, java.util.List.of());
        }

        static <T1, T2> LoggingMonad<T2> bind(
                LoggingMonad<T1> input,
                Function<T1, LoggingMonad<T2>> transform) {

            final LoggingMonad<T2> result = transform.apply(input.value);
            java.util.List<String> logs = new ArrayList<>(input.logs);
            logs.addAll(result.logs);
            return new LoggingMonad<>(result.value, logs);
        }

        static <T> LoggingMonad<T> pipeline(
                LoggingMonad<T> monad,
                java.util.List<Function<T, LoggingMonad<T>>> transforms) {

            LoggingMonad<T> result = monad;
            for (Function<T, LoggingMonad<T>> transform : transforms) {
                result = bind(result, transform);
            }
            return result;
        }
    }

    /**
     * Reader Monad
     *
     * Reader Monad 也被称为 Environment Monad，描述的是依赖共享环境的计算。
     * Reader Monad 的类型构造器从类型 T 中创建出一元类型 E → T，而 E 是环境的类型。
     * 类型构造器把类型 T 转换成一个从类型 E 到 T 的函数。Reader Monad 的 unit 操作把类型 T 的值 t 转换成一个永远返回 t 的函数，
     * 而忽略类型为 E 的参数；bind 操作在转换时，在所返回的函数的函数体中对类型 T 的值 t 进行转换，同时保持函数的结构不变。
     */
    /**
     * 清单 8 是 Reader Monad 的示例。Function<E, T> 是一元类型的声明。ReaderMonad 的 unit 方法返回的 Function 只是简单的返回参数值 value。
     * 而 bind 方法的第一个参数是一元类型 Function<E, T1>，第二个参数是把类型 T1 转换成 Function<E, T2> 的函数，
     * 返回值是另外一个一元类型 Function<E, T2>。bind 方法的转换逻辑首先通过 input.apply(e) 来得到类型为 T1 的值，
     * 再使用 transform.apply 来得到类型为 Function<E, T2>> 的值，最后使用 apply(e) 来得到类型为 T2 的值。
     */
    @Test
    public void readerMonad() {
        Function<Environment, String> m1 = ReaderMonad.unit("hello");
        Function<Environment, String> m2 = ReaderMonad.bind(m1, value -> e -> e.getPrefix() + value);
        Function<Environment, String> m3 = ReaderMonad.bind(m2, value -> e -> e.getBase() + "," + value);
        String result = m3.apply(new Environment());
        System.out.println(result);
    }

    private static class ReaderMonad {
        public static <T, E> Function<E, T> unit(T value) {
            return e -> value;
        }

        public static <T1, T2, E> Function<E, T2> bind(
                Function<E, T1> input,
                Function<T1, Function<E, T2>> transform) {
            return e -> transform.apply(input.apply(e)).apply(e);
        }
    }

    private class Environment {

        public String getPrefix() {
            return this.hashCode() + ",";
        }

        public long getBase() {
            return System.currentTimeMillis();
        }
    }

    /**
     * State Monad 可以在计算中附加任意类型的状态值。State Monad 与 Reader Monad 相似，
     * 只是 State Monad 在转换时会返回一个新的状态对象，从而可以描述可变的环境。
     * State Monad 的类型构造器从类型 T 中创建一个函数类型，该函数类型的参数是状态对象的类型 S，而返回值包含类型 S 和 T 的值。
     * State Monad 的 unit 操作返回的函数只是简单地返回输入的类型 S 的值；bind 操作所返回的函数类型负责在执行时传递正确的状态对象。
     */
    private static class StateMonad {
        public static <T, S> Function<S, Tuple2<T, S>> unit(T value) {
            return s -> Tuple.of(value, s);
        }

        public static <T1, T2, S> Function<S, Tuple2<T2, S>> bind(
                Function<S, Tuple2<T1, S>> input,
                Function<T1, Function<S, Tuple2<T2, S>>> transform) {
            return s -> {
                Tuple2<T1, S> result = input.apply(s);
                return transform.apply(result._1).apply(result._2);
            };
        }
    }

    @Data
    private static class State {
        private final int value;
    }

    @Test
    public void monad() {
        Function<String, Function<String, Function<State, Tuple2<String, State>>>> transform =
                prefix -> value -> s -> Tuple
                        .of(prefix + value, new State(s.getValue() + value.length()));
        Function<State,Tuple2<String,State>> m1 = StateMonad.unit("hello");
        Function<State,Tuple2<String,State>> m2 = StateMonad.bind(m1,transform.apply("1"));
        Function<State,Tuple2<String,State>> m3 = StateMonad.bind(m2,transform.apply("2"));
        Tuple2<String,State> result = m3.apply(new State(0));
        System.out.println(result);
    }
}

