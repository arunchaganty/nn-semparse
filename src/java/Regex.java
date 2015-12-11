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

  public void checkIfEqual(String s1, String s2) {
    RegExp r1 = new RegExp(s1);
    RegExp r2 = new RegExp(s2);
    Automaton a1 = r1.toAutomaton();
    Automaton a2 = r2.toAutomaton();
    System.out.println(s1 + " and " + s2 + " are equal? " + a1.equals(a2));
  }

  public void run() {
    checkIfEqual(".*", ".*.*");
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

