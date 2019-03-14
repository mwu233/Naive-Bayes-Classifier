/**
 * Classification result of a news article instance.
 * 
 * DO NOT MODIFY.
 */
public class ClassifyResult {
  /**
   * Sports or business
   */
  public Label label;
  /**
   * The log probability that the news article is about sports. Does not have to be normalized.
   */
  public double log_prob_sports;
  /**
   * The log probability that the news article is about business. Does not have to be normalized.
   */
  public double log_prob_business;
}
