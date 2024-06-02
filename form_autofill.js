// Define data for the form fields
const formData = {
    'v1': 0.1,
    'v2': 0.2,
    'v3': 0.3,
    'v4': 0.4,
    'v5': 0.5,
    'v6': 0.6,
    'v7': 0.7,
    'v8': 0.8,
    'v9': 0.9,
    'v10': 1.0,
    'v11': 1.1,
    'v12': 1.2,
    'v13': 1.3,
    'v14': 1.4,
    'v15': 1.5,
    'v16': 1.6,
    'v17': 1.7,
    'v18': 1.8,
    'v19': 1.9,
    'v20': 2.0,
    'v21': 2.1,
    'v22': 2.2,
    'v23': 2.3,
    'v24': 2.4,
    'v25': 2.5,
    'v26': 2.6,
    'v27': 2.7,
    'v28': 2.8,
    'amount': 1000 // Set amount value as needed
};

// Fill the form fields
Object.entries(formData).forEach(([fieldName, value]) => {
    document.getElementById(fieldName).value = value;
});