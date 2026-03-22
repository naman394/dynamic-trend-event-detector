import pandas as pd
import os

def process_gdelt(file_path):
    print(f"Processing GDELT file: {file_path}")
    
    # GDELT GKG uses tabs and has no header in raw form (usually)
    # Based on V2 spec: https://blog.gdeltproject.org/gdelt-2-0-our-global-knowledge-graph-gkg-v2-0-is-now-live/
    columns = [
        'GKGRECORDID', 'DATE', 'SOURCECOLLECTIONID', 'SOURCECOMMONNAME', 'DOCUMENTIDENTIFIER',
        'COUNTS', 'V2COUNTS', 'THEMES', 'V2THEMES', 'LOCATIONS', 'V2LOCATIONS', 'PERSONS', 'V2PERSONS',
        'ORGANIZATIONS', 'V2ORGANIZATIONS', 'TONE', 'V2TONE', 'ENHANCEDDATES', 'GCAM', 'SHARINGIMAGE',
        'RELATEDIMAGES', 'SOCIALIMAGEEMBEDS', 'SOCIALVIDEOEMBEDS', 'QUOTATIONS', 'ALLNAMES', 'AMOUNTS',
        'TRANSLATIONINFO', 'EXTRASXML'
    ]
    
    try:
        df = pd.read_csv(file_path, sep='\t', names=columns, encoding='utf-8')
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return None

    # Focus on THEMES and TONE
    # V2THEMES format: THEME,Offset;THEME,Offset...
    def extract_themes(v2themes):
        if pd.isna(v2themes): return []
        items = v2themes.split(';')
        themes = [item.split(',')[0] for item in items if item]
        return themes

    df['theme_list'] = df['V2THEMES'].apply(extract_themes)
    
    # Extract Tone (first value in TONE comma-separated string)
    def extract_tone(tone_str):
        if pd.isna(tone_str): return 0.0
        return float(tone_str.split(',')[0])

    df['tone_value'] = df['TONE'].apply(extract_tone)
    
    print(f"Extracted {len(df)} GDELT records.")
    return df[['GKGRECORDID', 'DATE', 'SOURCECOMMONNAME', 'DOCUMENTIDENTIFIER', 'theme_list', 'tone_value']]

if __name__ == "__main__":
    # Find the GKG file
    data_dir = 'data'
    gkg_files = [f for f in os.listdir(data_dir) if f.endswith('.gkg.csv')]
    if gkg_files:
        processed_df = process_gdelt(os.path.join(data_dir, gkg_files[0]))
        if processed_df is not None:
            processed_df.to_csv('data/gdelt_processed.csv', index=False)
            print("Saved processed GDELT data to data/gdelt_processed.csv")
    else:
        print("No GDELT GKG files found in data/")
