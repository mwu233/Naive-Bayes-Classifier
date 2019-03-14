
public class ConfusionMatrix {
  
  int TP, FP, FN, TN;
  
  public ConfusionMatrix(int TP, int FP, int FN, int TN) {
    this.TP = TP;
    this.FP = FP;
    this.FN = FN;
    this.TN = TN;
  }
  
  /*
   * Print out confusion matrix in readable format
   */
  @Override
  public String toString() {
    //                      True Sport     True Business
    //Classified Sport      TP             FP
    //Classified Business   FN             TN
    StringBuilder sb = new StringBuilder();
    sb.append("---------------------------------------------------------\n");
    sb.append("\t\t\t\t\t| True Sport\t| True Business\t|\n");
    sb.append("---------------------------------------------------------\n");
    sb.append("Classified Sport\t| ").append(TP).append("\t\t\t| ").append(FP).append("\t\t\t\t|\n");
    sb.append("Classified Business\t| ").append(FN).append("\t\t\t\t| ").append(TN).append("\t\t\t|\n");
    sb.append("---------------------------------------------------------");
    return sb.toString();
  }
  
  
}
