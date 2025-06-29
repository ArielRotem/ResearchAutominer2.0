import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Container, Grid, Box, Button, CircularProgress } from '@mui/material';
import DataPreview from './components/DataPreview';
import ManuscriptEditor from './components/ManuscriptEditor';
import LiveTestWindow from './components/LiveTestWindow';
import AIAssistant from './components/AIAssistant';
import axios from 'axios';

function App() {
  const [manuscript, setManuscript] = useState<any[]>([]);
  const [allData, setAllData] = useState<any[]>([]); // State to hold all loaded data
  const [sampleData, setSampleData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [runningFullManuscript, setRunningFullManuscript] = useState<boolean>(false);

  const handleRunFullManuscript = async () => {
    if (!allData.length) {
      alert("Please load a CSV file first.");
      return;
    }
    if (!manuscript.length) {
      alert("Please create a manuscript first.");
      return;
    }

    try {
      const response = await axios.post('http://localhost:8000/api/run_full_manuscript', {
        data: allData,
        manuscript: manuscript,
      });
      const processedData = response.data;
      // Trigger download of processedData
      const blob = new Blob([JSON.stringify(processedData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'processed_data.json'; // Or .csv if we convert it
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      alert("Manuscript executed and data downloaded!");
    } catch (error: any) {
      console.error("Error running full manuscript:", error);
      alert(`Error running full manuscript: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Research Autominer
          </Typography>
          <Button color="inherit" onClick={handleRunFullManuscript} disabled={runningFullManuscript}>
            {runningFullManuscript ? <CircularProgress size={24} color="inherit" /> : 'Run Full Manuscript'}
          </Button>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ mt: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <DataPreview allData={allData} setAllData={setAllData} setSampleData={setSampleData} setHeaders={setHeaders} headers={headers} />
          </Grid>
          
          <Grid item xs={12}>
            <LiveTestWindow manuscript={manuscript} sampleData={sampleData} headers={headers} />
          </Grid>

          <Grid item xs={6}>
            <ManuscriptEditor manuscript={manuscript} setManuscript={setManuscript} headers={headers} />
          </Grid>
          <Grid item xs={6}>
            <AIAssistant headers={headers} />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default App;
