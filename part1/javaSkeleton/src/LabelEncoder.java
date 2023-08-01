import java.util.*;

public class LabelEncoder {
    private final List<String> classes;

    public LabelEncoder(String[] labels) {
        classes = new ArrayList<>(new TreeSet<>(Arrays.asList(labels)));

    }

    public int[] intEncode(String[] labels) {
        int[] intLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            intLabels[i] = classes.indexOf(labels[i]);
        }
        return intLabels;
    }

    public int[][] oneHotEncode(String[] labels) {
        int[][] oneHotLabels = new int[labels.length][classes.size()];
        for (int i = 0; i < labels.length; i++) {
            int index = classes.indexOf(labels[i]);
            oneHotLabels[i][index] = 1;
        }
        return oneHotLabels;
    }

    public String inverse_transform(int i) {
        return classes.get(i);
    }
}
