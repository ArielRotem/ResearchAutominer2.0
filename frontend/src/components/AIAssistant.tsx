import React, { useState } from 'react';
import { Button, Paper, Typography, TextField, Box } from '@mui/material';
import Editor from '@monaco-editor/react';
import axios from 'axios';

interface AIAssistantProps {
  headers: string[];
}

const AIAssistant: React.FC<AIAssistantProps> = ({ headers }) => {
  const [prompt, setPrompt] = useState<string>("");
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleGenerate = async () => {
    setIsLoading(true);
    setGeneratedCode(""); // Clear previous code
    try {
      const response = await axios.post('http://localhost:8000/api/generate_function', {
        prompt,
        headers,
      });
      setGeneratedCode(response.data.code);
    } catch (error: any) {
      console.error("Error generating function:", error);
      setGeneratedCode(`Error generating function:\n${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 3 }}>
      <Typography variant="h6" gutterBottom>AI Function Generator</Typography>
      <TextField
        label="Describe the function you want to create"
        multiline
        rows={4}
        fullWidth
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        sx={{ mb: 2 }}
      />
      <Button variant="contained" onClick={handleGenerate} disabled={!prompt || !headers.length || isLoading}>
        {isLoading ? 'Generating...' : 'Generate Function'}
      </Button>
      {generatedCode && (
        <Box sx={{ mt: 2, height: '300px', border: '1px solid #ccc' }}>
          <Editor
            height="100%"
            defaultLanguage="python"
            value={generatedCode}
            options={{
              readOnly: true,
              minimap: { enabled: false },
              wordWrap: 'on',
              showUnused: false,
              folding: false,
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </Box>
      )}
    </Paper>
  );
};

export default AIAssistant;