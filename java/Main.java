import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        try {
            // 1. Display survey title
            System.out.println("=".repeat(50));
            System.out.println("Political Affiliation Survey");
            System.out.println("=".repeat(50));
            System.out.println();

            // 2. Load and parse questions from JSON file
            List<String> questions = parseQuestionsFromJson("../data/questions.json");

            // 3. Collect answers
            Scanner scanner = new Scanner(System.in);
            StringBuilder csvLine = new StringBuilder();

            for (String question : questions) {
                System.out.println(question); // print the question
                String answer = getValidAnswer(scanner, "Your answer (A/B/C/D): ");
                csvLine.append(answer).append(",");
            }

            // 3. Get validated party affiliation
            String party = getValidParty(scanner);
            csvLine.append(party);

            // 4. Save to CSV (create file if it doesn't exist)
            saveToFile(csvLine.toString());

            System.out.println("Thank you for completing the survey!");
            System.out.println("Run 'python train.py' to train the ML model.");
        } catch (NoSuchFileException e) {
            System.err.println("Error: The file does not exist. Please check the path.");
        } catch (IOException e) {
            System.err.println("Error reading or writing to file: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // All helper methods must be INSIDE the class
    private static List<String> parseQuestionsFromJson(String filePath) throws IOException {
        String jsonContent = Files.readString(Paths.get(filePath));
        List<String> questions = new ArrayList<>();
        
        // Simple JSON parsing - extract questions
        String[] questionBlocks = jsonContent.split("\"question\":");
        
        for (int i = 1; i < questionBlocks.length; i++) {
            String questionBlock = questionBlocks[i];
            
            // Extract question text
            int start = questionBlock.indexOf("\"") + 1;
            int end = questionBlock.indexOf("\"", start);
            String questionText = questionBlock.substring(start, end);
            
            // Extract options
            String optionsText = questionBlock.substring(questionBlock.indexOf("\"options\""));
            String[] optionBlocks = optionsText.split("\"text\":");
            
            StringBuilder formattedQuestion = new StringBuilder();
            formattedQuestion.append(i).append(". ").append(questionText).append("\n");
            
            for (int j = 1; j < optionBlocks.length && j <= 4; j++) {
                String optionBlock = optionBlocks[j];
                int optStart = optionBlock.indexOf("\"") + 1;
                int optEnd = optionBlock.indexOf("\"", optStart);
                String optionText = optionBlock.substring(optStart, optEnd);
                
                char letter = (char)('A' + j - 1);
                formattedQuestion.append("   ").append(letter).append(") ").append(optionText).append("\n");
            }
            formattedQuestion.append("\n");
            
            questions.add(formattedQuestion.toString());
        }
        
        return questions;
    }

    private static String getValidAnswer(Scanner scanner, String prompt) {
        while (true) {
            System.out.print(prompt);
            String input = scanner.nextLine().trim().toUpperCase();
            if (input.matches("^[ABCD]$")) {
                return input;
            } else {
                System.out.println("Invalid input. Please enter A, B, C, or D.");
            }
        }
    }

    private static String getValidParty(Scanner scanner) {
        while (true) {
            System.out.print("Please enter your party affiliation (Progressive/Conservative/Libertarian/Moderate): ");
            String party = scanner.nextLine().trim().toLowerCase();
            if (party.equals("progressive") 
            || party.equals("conservative") 
            || party.equals("libertarian")
            || party.equals("moderate")) {
                return party;
            } else {
                System.out.println("Invalid input. Please enter Progressive, Conservative, Libertarian, or Moderate.");
            }
        }
    }

    private static void saveToFile(String csvLine) throws IOException {
        Path filePath = Paths.get("../data/survey_results.csv");
        if (!Files.exists(filePath)) {
            Files.createFile(filePath);
        }
        Files.write(filePath, Collections.singleton(csvLine), StandardOpenOption.APPEND);
    }
}