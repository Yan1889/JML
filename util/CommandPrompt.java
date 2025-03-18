package libs.JML.util;


import libs.JML.Models.PredictionModel;
import libs.JML.Models.Regressor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/// available commands: test, save, quit
public class CommandPrompt {
    public static void waitAndHandleCommand(PredictionModel model) throws IOException {
        Scanner sc = new Scanner(System.in);

        System.out.print("~> ");
        String command = sc.nextLine();

        switch (command) {
            case "test" -> testOnUser(model);
            case "save" -> {
                System.out.print("what should the file be called? ~> ");
                String fileName = sc.nextLine();
                model.writeToFile(fileName);
                System.out.println("written to file '" + fileName + "' successfully");
            }
            case "quit" -> {
                System.out.println("manual quit");
                System.exit(0);
            }
            default -> System.out.println("command '" + command + "' not found, commands: {save, test, quit}");
        }
    }

    private static void testOnUser(PredictionModel model) {
        List<Double> customInputs = new ArrayList<>();

        Scanner sc = new Scanner(System.in);

        for (int i = 0; i < ((Regressor) model).layerSizes.getFirst(); i++) {
            System.out.print("Enter " + i + ". argument: ");
            customInputs.add(sc.nextDouble());
            System.out.println("your arg: " + customInputs.getLast());
        }

        System.out.println("answer: " + model.predict(customInputs));
    }
}
