import pandas as pd
import numpy as np

class DataIngestionAgent:
    """
    Data Loading, Sanitization & Feature Engineering Agent.
    Responsible for ingest pipelines, string normalization, and building model matrices.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self) -> pd.DataFrame:
        """Loads and converts the raw dataset into an analytics-ready DataFrame."""
        try:
            # Explicit handling of potential encoding quirks often found in Russian municipal data
            df = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_path, encoding='cp1251')

        # Drop any structural rows that are totally missing critical target figures
        df = df.dropna(subset=['Количество видеокамер', 'Район', 'Адрес'])
        
        # Enforce exact system types
        df['Количество видеокамер'] = df['Количество видеокамер'].astype(int)
        df['Район'] = df['Район'].astype(str).str.strip()
        df['Адрес'] = df['Адрес'].astype(str).str.strip()

        # Feature Extraction: Classify administrative layout profiles based on keyword mapping
        df['Street_Type'] = df['Адрес'].apply(self._extract_street_type)
        
        # Structural Feature: Text string metadata length acts as a statistical proxy for density matrix
        df['Address_Length'] = df['Адрес'].apply(lambda x: len(str(x)))
        
        return df

    @staticmethod
    def _extract_street_type(address: str) -> str:
        """Categorizes raw structural addresses into localized urban infrastructure bins."""
        addr_lower = address.lower()
        if 'улица' in addr_lower: 
            return 'Улица (Street)'
        elif 'проспект' in addr_lower: 
            return 'Проспект (Avenue)'
        elif 'набережная' in addr_lower: 
            return 'Набережная (Embankment)'
        elif 'шоссе' in addr_lower: 
            return 'Шоссе (Highway)'
        elif 'переулок' in addr_lower: 
            return 'Переулок (Lane)'
        elif 'линия' in addr_lower: 
            return 'Линия (Line)'
        elif 'бульвар' in addr_lower:
            return 'Бульвар (Boulevard)'
        else: 
            return 'Другое (Other Categories)'
