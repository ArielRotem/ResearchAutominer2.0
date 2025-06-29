
import React, { useState } from 'react';
import { Button, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Grid, CircularProgress } from '@mui/material';
import axios from 'axios';

interface LiveTestWindowProps {
  manuscript: any[];
  sampleData: any[];
  headers: string[];
}

const LiveTestWindow: React.FC<LiveTestWindowProps> = ({ manuscript, sampleData, headers }) => {
  const [outputData, setOutputData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [testingManuscript, setTestingManuscript] = useState<boolean>(false);

  const handleTest = async () => {
    setError(null); // Clear previous errors
    try {
      const response = await axios.post('http://localhost:8000/api/test_manuscript', {
        manuscript,
        sample_data: sampleData,
      });
      setOutputData(response.data);
    } catch (err: any) {
      console.error("Error testing manuscript:", err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError("An unknown error occurred during manuscript testing.");
      }
      setOutputData([]); // Clear output data on error
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>Live Data Test</Typography>
      <Button variant="contained" onClick={handleTest} disabled={!manuscript.length || !sampleData.length || testingManuscript} sx={{ mb: 2 }}>
        {testingManuscript ? <CircularProgress size={24} /> : 'Test'}
      </Button>
      {error && (
        <Typography color="error" sx={{ mt: 2, whiteSpace: 'pre-wrap' }}>
          Error: {error}
        </Typography>
      )}
      <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <Typography variant="subtitle1">Input</Typography>
          <TableContainer sx={{ maxHeight: 300, borderRadius: '8px', border: '1px solid #e0e0e0' }}>
            <Table stickyHeader size="small" sx={{ '& .MuiTableCell-root': { borderBottom: '1px solid #e0e0e0' } }}>
              <TableHead>
                <TableRow>
                  {headers.map((header) => (
                    <TableCell key={header}>{header}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {sampleData.slice(0, 5).map((row, index) => (
                  <TableRow key={index}>
                    {headers.map((header) => (
                      <TableCell key={header}>{row[header]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="subtitle1">Output</Typography>
          <TableContainer sx={{ maxHeight: 300, borderRadius: '8px', border: '1px solid #e0e0e0' }}>
            <Table stickyHeader size="small" sx={{ '& .MuiTableCell-root': { borderBottom: '1px solid #e0e0e0' } }}>
              <TableHead>
                <TableRow>
                  {headers.map((header) => (
                    <TableCell key={header}>{header}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {outputData.map((row, index) => (
                  <TableRow key={index}>
                    {headers.map((header) => (
                      <TableCell key={header}>{row[header]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default LiveTestWindow;
