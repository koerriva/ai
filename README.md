# ai

https://www.cnblogs.com/hesi/p/7218602.html

```Java
public class NeuralNetwork {
    private Logger logger = LoggerFactory.getLogger(NeuralNetwork.class);

    private int[] layerShape;//{2, 2, 1} 三层结构
    private double[] weights;//6个权重
    private double[] outputs;//

    private int weightNum;//权重个数

    public NeuralNetwork(double[] weights,int[] layerShape) {
        this.layerShape = layerShape;
        this.weights = weights;

        logger.info("输入:{}",layerShape[0]);
        logger.info("输出:{}",layerShape[layerShape.length-1]);

        for (int i = 1; i < layerShape.length; i++) {
            weightNum +=layerShape[i]*layerShape[i-1];
            weightNum +=1;
        }

        logger.info("权重数:{}",weightNum);

        if(weightNum!=weights.length){
            logger.error("权重数不匹配！！！");
        }
    }

    //三层网络
    public void run(double[] inputs){
        //第二层
        //每个输出 = 前一层输入*权重
        outputs = new double[layerShape[1]];
        for (int i = 0; i < layerShape[1] ; i++) {
            for (int j = 0; j < layerShape[0]; j++) {
                outputs[i] +=inputs[j]*weights[i*layerShape[0]+j];
            }
            double b = weights[layerShape[1]*layerShape[0]];
            logger.info("b:{}",b);
            outputs[i] = sigmod(outputs[i]+b);
        }
        logger.info("outputs:{}",outputs);
        //第三层
        inputs = new double[layerShape[1]];
        System.arraycopy(outputs, 0, inputs, 0, outputs.length);
        outputs = new double[layerShape[2]];
        for (int i = 0; i < layerShape[2]; i++) {
            for (int j = 0; j < layerShape[1]; j++) {
                outputs[i] +=inputs[j]*weights[i*layerShape[1]+j + layerShape[1]*layerShape[0] + 1];
            }
            double b = weights[layerShape[2]*layerShape[1] + layerShape[1]*layerShape[0] + 1];
            logger.info("b:{}",b);
            outputs[i] = sigmod(outputs[i]+b);
        }
        logger.info("outputs:{}",outputs);

        for (int l = 1; l < layerShape.length; l++) {
            outputs = new double[layerShape[l]];
            for (int i = 0; i < layerShape[l]; i++) {
                for (int j = 0; j < layerShape[l-1]; j++) {
                    outputs[i] +=inputs[j]*weights[i*layerShape[l-1]+j];
                }
                double b = weights[layerShape[l]*layerShape[l-1]];
                logger.info("b:{}",b);
                outputs[i] = sigmod(outputs[i]+b);
            }
            logger.info("outputs:{}",outputs);
        }
    }

    public void update(double[] weights){
        this.weights = weights;
    }

    private double sum(double[] inputs, double[] weights) {
        double sum = 1;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    private double sigmod(double x){
        return 1/(1+Math.exp(-x));
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "weights=" + Arrays.toString(weights) +
                ", outputs=" + Arrays.toString(outputs) +
                '}';
    }
}
```
