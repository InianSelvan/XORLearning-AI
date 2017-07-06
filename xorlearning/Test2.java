package xorlearning;

public class Test2 {

	public static void main(String[] args) {
		int temp = 0;
		//for(int k =0 ; k< 10 ; k++) {
		
		//used three input and 5 hidden neurons to manage bias
		//The program manages to hard code +1 for all calculation, this is just for my convenience 
		
		//the final argument is true which means bipolar, false binary
		
			NeuralNetInterfaceImpl NN = new NeuralNetInterfaceImpl(3, 5, 0.2, 0.9, -1, 1, true);
			
			double x[][] = {
							{-1,-1,+1},
							{-1,+1,+1},
							{+1,-1,+1},
							{+1,+1,+1}
							};
			double y[] = {-1,1,1,-1};
			
			for(int i = 0; i < 1000; i++) {
				double forEachStep = 0;
				//System.out.println("\n");
				for(int j = 0; j < 4; j++) {
					forEachStep = forEachStep + NN.train(x[j], y[j]);
					
				}
				double Error = 0.5 * forEachStep;
				System.out.println("Epoch -"+i + "-ERROR- " +Error);
				if(Error <= 0.05) {
					System.out.println("ERROR- " +i);
					temp = temp + i;
					break;
				}
			}

		//}
		//System.out.println("Average - "+ temp/10);
	}

}
