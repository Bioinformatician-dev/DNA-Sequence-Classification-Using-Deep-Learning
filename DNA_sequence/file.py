import pandas as pd

# Sample data
data = {
    'sequence': [
        'ATGCGT', 'ATGCTA', 'ATGCGT', 'ATGCAT', 'ATGGTA', 
        'CTAGCT', 'CTAGAT', 'CTAGGT', 'CTAGTA', 'ATGCGC',
        'ATGCCC', 'CTAGGA', 'ATGGTG', 'CTAGAC', 'CTAGTA'
    ],
    'label': [
        'cancerous', 'non-cancerous', 'cancerous', 'non-cancerous', 
        'cancerous', 'non-cancerous', 'non-cancerous', 'cancerous', 
        'non-cancerous', 'cancerous', 'non-cancerous', 'cancerous', 
        'cancerous', 'non-cancerous', 'non-cancerous'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('dna_sequences.csv', index=False)
