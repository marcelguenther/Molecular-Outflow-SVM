# Molecular-Outflow-SVM
Developing of a Support Vector Machine to detect Molecular Outflow in Star Forming Regions

# Outflow Detection with SVM  

This repository provides tools for applying a pre-trained Support Vector Machine (SVM) model to identify molecular outflows in astronomical data cubes. Below are instructions for using the model and performing initial analyses of the results.  

## **Getting Started**  

### **Step 1: Prepare the Input Cube**  
- Add the FITS file of the cube you want to analyze to the `SVM-Cubes.csv` file.  
  - Ensure the cube is in **velocity space**.  

### **Step 2: Configure `SVMMaster.xml`**  
- Open the `SVMMaster` configuration file and verify the following settings:  
  1. Path to the `SVM-Cubes.csv` file.  
  2. Path to the pre-trained model (`Outflow_SVM.pkl`).  
  3. Define the following paths in the XML structure:  
     ```xml
     <!-- Root in which all files/directories will be stored -->
     <SavePath>/SavePath/</SavePath>
     <!-- Directory from which most files will be read -->
     <ReadPath>/ReadPath/</ReadPath>
     <!-- Directory to the database(s) -->
     <DatabasePath>/DatabasePath/</DatabasePath>
     <!-- Directory containing the trained SVM -->
     <Read_SVM_Path>/SVMPath/</Read_SVM_Path>
     ```  
  - Ensure that all necessary files and directories are stored and referenced correctly according to these paths.  

- **Note:** For direct application, do not modify settings related to training, validation, or additional outputs.  

### **Step 3: Run the SVM Analysis**  
- Open `Main_SVM.py` and verify that the paths are set correctly.  
- Execute `Main_SVM.py` to start the analysis.  

### **Output**  
- The script will generate a file named `Predict` in the **Output** folder.  
  - This file has the same shape as the input cube.  
  - **Predictions:**  
    - `1` → Indicates the presence of an outflow.  
    - `0` → Indicates no outflow.  

---

## **Analyzing Results**  

### **Step 4: Perform Basic Analyses**  
- Add the `Predict` file and the corresponding cube to `Cubes-Info.csv`.  
- Use the following scripts for further analysis:  
  1. **`full_outflow_analysis`**: For detailed evaluation of the cube and prediction results.  
  2. **`volume_analysis`**: For volumetric analysis of detected outflows.  

### **Step 5: Extended Analysis (Optional)**  
- Advanced analysis tools for processing multiple cubes will be provided in future updates.  

---

## **Folder and File Structure**  
```plaintext
├── SVM-Cubes.csv          # List of input cubes to analyze
├── SVMMaster              # Configuration file for SVM paths and settings
├── Main_SVM.py            # Main script for applying the SVM model
├── Outflow_SVM.pkl        # Pre-trained SVM model
├── Cubes-Info.csv         # File for linking cubes and results for further analysis
├── Scripts/               # Additional analysis scripts (e.g., `full_outflow_analysis`, `volume_analysis`)
└── Output/                # Folder containing the `Predict` output file (can be located elsewhere)
