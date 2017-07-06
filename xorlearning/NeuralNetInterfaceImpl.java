package xorlearning;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NeuralNetInterfaceImpl implements NeuralNetInterface{
	
	private double[][] inputToHiddenNeuronWeights;
	private double[] hiddenNeuralNetOutput;
	private double[] hiddenNeuronToOutputWeights;
	private double argA = 0;
	private double argB = 0;
	private int argNumHidden = 0;
	private double argLearningRate;
	private int argNumInputs = 0;
	private boolean isBipolar;
	
	private double argMomentumTerm;
	private double[][] inputToHiddenNeuronPreviousWeights;
	private double[] hiddenNeuronToOutputPreviousWeights;
	
	public NeuralNetInterfaceImpl(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA, double argB, boolean isBipolar ) {
		
		this.argA = argA;
		this.argB = argB;
		this.argLearningRate = argLearningRate;
		this.argNumHidden =argNumHidden;
		this.argNumInputs = argNumInputs;
		this.isBipolar = isBipolar;
		this.argMomentumTerm =argMomentumTerm;
		
		inputToHiddenNeuronWeights = new double[argNumHidden][argNumInputs];
		inputToHiddenNeuronPreviousWeights = new double[argNumHidden][argNumInputs];
		hiddenNeuralNetOutput = new double[argNumHidden];
		hiddenNeuronToOutputWeights = new double[argNumHidden];
		hiddenNeuronToOutputPreviousWeights = new double[argNumHidden];
		initializeWeights();
		zeroWeights();
		
	}

	@Override
	public double outputFor(double[] X) {
		double uij = 0;
		double output = 0;
		
		//finding the outputs of all the hidden neurons uij 
		// ∑ (Wij . ui)
		
		for(int i=0; i<argNumHidden; i++) {	
			for(int j=0; j < X.length; j++) {																			
				uij = uij + inputToHiddenNeuronWeights[i][j] * X[j] ;
			}
		
			hiddenNeuralNetOutput[i] = sigmoid(uij);	
			if(i == (argNumHidden - 1)) { 
				hiddenNeuralNetOutput[i] =  1;
				}
			//finding the final output which is again
			// ∑ (Wij . ui)			
			output = output +  ( hiddenNeuronToOutputWeights[i] * hiddenNeuralNetOutput[i] );
		}
		
		//using this value of output to sigmoid
		// which is 1/(1 + e^-output)
		return sigmoid(output);
	}

	@Override
	public double train(double[] X, double argValue) {
		
		double uj = outputFor(X);
		double outputsDerivative;
		double hiddenDerivative;
		
		if(isBipolar) {
			outputsDerivative = 0.5 * (1 - Math.pow(uj, 2));
		}else {
			outputsDerivative = uj * (1 - uj);
		}
		
		//Back Propagation
		//finding the output error signal δi =( Ci −ui )f'(Si)
		
		double OutputErrorSignal = (argValue - uj) * outputsDerivative;
		double hiddenErrorSignal[] = new double[argNumHidden];
		
		
		//updating the weights for output to hidden layer neurons 
		//since we have the outputs error signal δi it is easy to calculate the weights
		// Wij = Wij + ρ.δi.u (updating all hidden neuron to output weights)
		
		for(int i = 0; i < hiddenNeuronToOutputWeights.length; i++) {			

				hiddenNeuronToOutputWeights[i] = hiddenNeuronToOutputWeights[i] + 
												(hiddenNeuralNetOutput[i] * OutputErrorSignal *  argMomentumTerm )+
												(argLearningRate * OutputErrorSignal * hiddenNeuralNetOutput[i]);
		}
		
		//error signal for hidden layers δij
		//  f'(Si)  (Wij . δi )
		//  f'(Si) is for each neuron 
		for(int i = 0; i < argNumHidden-1; i++) {
			if(isBipolar) {
				hiddenDerivative = 0.5 * ( 1 - Math.pow(hiddenNeuralNetOutput[i],2));
			}else {
				hiddenDerivative = hiddenNeuralNetOutput[i] * ( 1 - hiddenNeuralNetOutput[i]);	
			}
			
			hiddenErrorSignal[i] = hiddenDerivative * hiddenNeuronToOutputWeights[i] * OutputErrorSignal;
		}
		
		//updating the weights for hidden to input layer weights
		//Wij = Wij + ρ.δi.u
		for(int i = 0; i < argNumHidden-1; i++) {
			for(int j = 0; j < X.length; j++) {
	
					inputToHiddenNeuronWeights[i][j] = inputToHiddenNeuronWeights[i][j] +
														(X[j] * hiddenErrorSignal[i] *argMomentumTerm) + 
														(argLearningRate * hiddenErrorSignal[i] * X[j]);
			
			}			
		}		
		
		return Math.pow((uj - argValue), 2);
	}

	@Override
	public void save(File argFile) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void load(String argFileName) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double sigmoid(double x) {
		
		if(isBipolar)
			return (1-Math.exp(-x)) / (1+Math.exp(-x));
		else
			return 1/(1 + Math.exp(-x));
	}

	@Override
	public double customSigmoid(double x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void initializeWeights() {
		Random rdn = new Random();
		for(int i=0; i<argNumHidden; i++) {	
			for(int j=0; j < argNumInputs; j++) {			
				inputToHiddenNeuronWeights[i][j] = rdn.nextDouble()*(argB - argA) + argA;
			}
			hiddenNeuronToOutputWeights[i] = (rdn.nextDouble()*(argB - argA)) + argA;
			
		}				
	}

	@Override
	public void zeroWeights() {
		
		for(int i=0; i<argNumHidden; i++) {	
			for(int j=0; j < argNumInputs; j++) {			
				inputToHiddenNeuronPreviousWeights[i][j] = inputToHiddenNeuronWeights[i][j];
			}
			hiddenNeuronToOutputPreviousWeights[i] = hiddenNeuronToOutputWeights[i];
			
		}
		
		
	}

}
