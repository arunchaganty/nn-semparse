package evaluator;

import dk.brics.automaton.Automaton;
import dk.brics.automaton.RegExp;
import java.util.*;

/**
 * Given two regexes, check if they are equivalent.
 *
 * @author Robin Jia
 */
public class Regex implements Runnable {
  public static String regex1;
  public static String regex2;

  public Automaton makeAutomaton(String s) {
    try {
      RegExp r = new RegExp(s);
      Automaton a = r.toAutomaton();
      return a;
    } catch (Exception e) {
      e.printStackTrace();
    } catch (OutOfMemoryError e) {
      e.printStackTrace();
    }
    return null;
  }

  public void checkIfEqual(String s1, String s2) {
    Automaton a1 = makeAutomaton(s1);
    Automaton a2 = makeAutomaton(s2);
    if (a1 == null) {
      String msg = "Error converting argument 1 = " + s1 + " to automaton";
      System.out.println(msg);
      System.err.println(msg);
    } else if (a2 == null) {
      String msg = "Error converting argument 2 = " + s2 + " to automaton";
      System.out.println(msg);
      System.err.println(msg);
    } else {
      System.out.println(s1 + " and " + s2 + " are equal? " + a1.equals(a2));
    }
  }

  public void run() {
    checkIfEqual(regex1, regex2);
  }

  public static void main(String[] args) {
    if (args.length != 2) {
      System.err.println("Received " + args.length + " != 2 arguments");
      System.err.println("Try surrounding regexes in parentheses " +
                         "to prevent wildcard expansion.");
      throw new RuntimeException("Expected args: [regex1] [regex2]");
    }
    regex1 = args[0];
    regex2 = args[1];

    new Regex().run();
  }
}

