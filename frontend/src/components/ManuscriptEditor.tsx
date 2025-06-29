import React, { useState, useEffect } from 'react';
import { Button, Paper, Typography, List, ListItem, ListItemText, Select, MenuItem, FormControl, InputLabel, TextField, Box, IconButton, Tooltip, Switch, FormControlLabel, CircularProgress } from '@mui/material';
import { AddCircleOutline, RemoveCircleOutline, Info as InfoIcon } from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import axios from 'axios';

interface ManuscriptEditorProps {
  manuscript: any[];
  setManuscript: (manuscript: any[]) => void;
  headers: string[];
}

const ManuscriptEditor: React.FC<ManuscriptEditorProps> = ({ manuscript, setManuscript, headers }) => {
  const [functions, setFunctions] = useState<any>({});
  const [selectedFunction, setSelectedFunction] = useState<string>("");
  const [manuscriptName, setManuscriptName] = useState<string>("");
  const [savedManuscripts, setSavedManuscripts] = useState<string[]>([]);
  const [loadingFunctions, setLoadingFunctions] = useState<boolean>(true);
  const [loadingManuscripts, setLoadingManuscripts] = useState<boolean>(true);
  const [savingManuscript, setSavingManuscript] = useState<boolean>(false);
  const [loadingSpecificManuscript, setLoadingSpecificManuscript] = useState<boolean>(false);
  const [renderedStepsCount, setRenderedStepsCount] = useState<number>(0);

  useEffect(() => {
    const fetchFunctions = async () => {
      setLoadingFunctions(true);
      try {
        const response = await axios.get('http://localhost:8000/api/functions');
        const funcs: Record<string, any> = {};
        for (const key in response.data as Record<string, any>) {
          const funcData = response.data[key];
          funcs[key] = {
            name: funcData.name,
            doc: funcData.doc,
            params: funcData.params.reduce((acc: Record<string, any>, param: any) => {
              acc[param.name] = param; // Store params by name for easy lookup
              return acc;
            }, {})
          };
        }
        setFunctions(funcs);
      } catch (error) {
        console.error("Error fetching functions:", error);
      } finally {
        setLoadingFunctions(false);
      }
    };
    const fetchManuscripts = async () => {
      setLoadingManuscripts(true);
      try {
        const response = await axios.get('http://localhost:8000/api/manuscripts');
        setSavedManuscripts(response.data);
      } catch (error) {
        console.error("Error fetching manuscripts:", error);
      } finally {
        setLoadingManuscripts(false);
      }
    };
    fetchFunctions();
    fetchManuscripts();
  }, []);

  useEffect(() => {
    if (manuscript.length > 0 && renderedStepsCount < manuscript.length) {
      const timer = setTimeout(() => {
        setRenderedStepsCount((prevCount: number) => Math.min(prevCount + 5, manuscript.length));
      }, 100); // Render 1 step every 100ms
      return () => clearTimeout(timer);
    }
  }, [manuscript, renderedStepsCount]);

  const handleAddStep = () => {
    if (selectedFunction && functions[selectedFunction]) {
      const newStep: { name: string; params: Record<string, any> } = {
        name: selectedFunction,
        params: {},
      };
      // Initialize params with default values if they exist
      for (const paramName in functions[selectedFunction].params) {
        const param = functions[selectedFunction].params[paramName];
        if (param.default !== null) {
          newStep.params[paramName] = param.default;
        }
      }
      setManuscript([...manuscript, newStep]);
    }
  };

  const handleParamChange = (stepIndex: number, paramName: string, value: any) => {
    const newManuscript = [...manuscript];
    newManuscript[stepIndex].params[paramName] = value;
    setManuscript(newManuscript);
  };

  const handleSave = async () => {
    try {
      await axios.post(`http://localhost:8000/api/manuscripts/${manuscriptName}`, manuscript);
      alert("Manuscript saved!");
      const response = await axios.get('http://localhost:8000/api/manuscripts');
      setSavedManuscripts(response.data);
    } catch (error) {
      console.error("Error saving manuscript:", error);
      alert("Error saving manuscript.");
    }
  };

  const handleLoad = async (name: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/manuscripts/${name}`);
      setManuscript(response.data);
      setManuscriptName(name);
    } catch (error) {
      console.error("Error loading manuscript:", error);
      alert("Error loading manuscript.");
    }
  };

  const renderParamInput = (step: any, stepIndex: number, paramName: string) => {
    // Hide the 'data' parameter as it's implicitly handled
    if (paramName === 'data') return null;

    const param = functions[step.name]?.params[paramName];
    if (!param) return null;

    let paramValue = step.params[paramName];
    // Ensure paramValue is not undefined for controlled components and handle dict/list defaults
    if (paramValue === undefined || paramValue === null) {
      if (param.annotation && (param.annotation === "<class 'dict'>" || param.annotation.includes('dict'))) {
        paramValue = {};
      } else if (param.annotation && param.annotation.includes('list')) {
        paramValue = [];
      } else {
        // Handle typing.Any with numeric defaults
        if (param.annotation && param.annotation.includes('typing.Any') && (param.default === 0 || param.default === 1)) {
          paramValue = param.default;
        } else {
          paramValue = param.default !== null ? param.default : '';
        }
      }
    }

    // Boolean parameters
    if (param.annotation === 'bool') {
      return (
        <FormControlLabel
          key={param.name}
          control={
            <Switch
              checked={!!paramValue}
              onChange={(e) => handleParamChange(stepIndex, param.name, e.target.checked)}
            />
          }
          label={param.name}
          sx={{ mt: 1 }}
        />
      );
    }

    // Parameters that are single column names
    if (param.annotation === 'str' && (param.name.includes('col') || param.name.includes('column') || param.name.includes('name') || param.name.includes('source') || param.name.includes('target') || param.name.includes('result')) && !param.name.includes('list') && !param.name.includes('dict')) {
      return (
        <FormControl fullWidth sx={{ mt: 1 }} size="small">
          <InputLabel>{param.name}</InputLabel>
          <Select
            value={String(paramValue) || ''}
            label={param.name}
            onChange={(e) => handleParamChange(stepIndex, param.name, e.target.value)}
          >
            {headers.map((header) => (
              <MenuItem key={header} value={header}>
                {header}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
    }

    // Parameters that are lists of strings (e.g., words, keywords, phrases)
    if (param.annotation && (param.annotation.includes('list[str]') || param.annotation.includes('list')) && !param.annotation.includes('dict')) {
      const listValues = Array.isArray(paramValue) ? paramValue : (paramValue ? [paramValue] : []);
      return (
        <Box key={param.name} sx={{ mt: 1, border: '1px solid #ccc', p: 1, borderRadius: '4px' }}>
          <Typography variant="subtitle2">{param.name} (List)</Typography>
          {listValues.map((item: string, itemIndex: number) => (
            <Box key={itemIndex} sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
              <TextField
                fullWidth
                size="small"
                value={item}
                onChange={(e) => {
                  const newList = [...listValues];
                  newList[itemIndex] = e.target.value;
                  handleParamChange(stepIndex, param.name, newList);
                }}
              />
              <IconButton size="small" onClick={() => {
                const newList = listValues.filter((_, i) => i !== itemIndex);
                handleParamChange(stepIndex, param.name, newList);
              }}>
                <RemoveCircleOutline />
              </IconButton>
            </Box>
          ))}
          <Button size="small" startIcon={<AddCircleOutline />} onClick={() => {
            handleParamChange(stepIndex, param.name, [...listValues, '']);
          }}>
            Add Item
          </Button>
        </Box>
      );
    }

    // Parameters that are dictionaries
    if (param.annotation && param.annotation.startsWith('dict')) {
      const dictValues = paramValue || {};
      const isValueList = param.annotation.includes('list'); // Heuristic for dict values being lists

      return (
        <Box key={param.name} sx={{ mt: 1, border: '1px solid #ccc', p: 1, borderRadius: '4px' }}>
          <Typography variant="subtitle2">{param.name} (Dictionary)</Typography>
          {Object.entries(dictValues).map(([key, value], itemIndex) => (
            <Box key={itemIndex} sx={{ display: 'flex', alignItems: 'center', mt: 0.5, gap: 1 }}>
              <TextField
                label="Key"
                size="small"
                value={key}
                onChange={(e) => {
                  const newDict = { ...dictValues };
                  const oldKey = Object.keys(dictValues)[itemIndex];
                  delete newDict[oldKey];
                  newDict[e.target.value] = value;
                  handleParamChange(stepIndex, param.name, newDict);
                }}
                sx={{ flex: 1 }}
              />
              {isValueList ? (
                <Box sx={{ flex: 2, border: '1px solid #eee', p: 0.5, borderRadius: '4px' }}>
                  <Typography variant="caption">Value (List)</Typography>
                  {(Array.isArray(value) ? value : [String(value)]).map((listItem: string, listIndex: number) => (
                    <Box key={listIndex} sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <TextField
                        fullWidth
                        size="small"
                        value={listItem}
                        onChange={(e) => {
                          const newValues = [...(Array.isArray(value) ? value : [String(value)])];
                          newValues[listIndex] = e.target.value;
                          const newDict = { ...dictValues, [key]: newValues };
                          handleParamChange(stepIndex, param.name, newDict);
                        }}
                      />
                      <IconButton size="small" onClick={() => {
                        const newValues = (Array.isArray(value) ? value : [String(value)]).filter((_, i) => i !== listIndex);
                        const newDict = { ...dictValues, [key]: newValues };
                        handleParamChange(stepIndex, param.name, newDict);
                      }}>
                        <RemoveCircleOutline />
                      </IconButton>
                    </Box>
                  ))}
                  <Button size="small" startIcon={<AddCircleOutline />} onClick={() => {
                    const newValues = [...(Array.isArray(value) ? value : [String(value)])];
                    newValues.push('');
                    const newDict = { ...dictValues, [key]: newValues };
                    handleParamChange(stepIndex, param.name, newDict);
                  }}>
                    Add Value
                  </Button>
                </Box>
              ) : (
                <TextField
                  label="Value"
                  size="small"
                  value={String(value)}
                  onChange={(e) => {
                    const newDict = { ...dictValues, [key]: e.target.value };
                    handleParamChange(stepIndex, param.name, newDict);
                  }}
                  sx={{ flex: 2 }}
                />
              )}
              <IconButton size="small" onClick={() => {
                const newDict = { ...dictValues };
                delete newDict[key];
                handleParamChange(stepIndex, param.name, newDict);
              }}>
                <RemoveCircleOutline />
              </IconButton>
            </Box>
          ))}
          <Button size="small" startIcon={<AddCircleOutline />} onClick={() => {
            const newDict = { ...dictValues, new_key: isValueList ? [''] : '' };
            handleParamChange(stepIndex, param.name, newDict);
          }}>
            Add Entry
          </Button>
        </Box>
      );
    }

    // Default to TextField for other parameters, with type="number" for numeric annotations
    const isNumeric = param.annotation && (param.annotation.includes('int') || param.annotation.includes('float'));
    return (
      <TextField
        key={param.name}
        label={param.name}
        fullWidth
        variant="outlined"
        size="small"
        sx={{ mt: 1 }}
        value={String(paramValue) || ''}
        onChange={(e) => handleParamChange(stepIndex, param.name, isNumeric ? Number(e.target.value) : e.target.value)}
        type={isNumeric ? 'number' : 'text'}
      />
    );
  };

  const onDragEnd = (result: any) => {
    if (!result.destination) {
      return;
    }

    const reorderedManuscript = Array.from(manuscript);
    const [removed] = reorderedManuscript.splice(result.source.index, 1);
    reorderedManuscript.splice(result.destination.index, 0, removed);

    setManuscript(reorderedManuscript);
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>Manuscript Editor</Typography>
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <TextField
          label="Manuscript Name"
          value={manuscriptName}
          onChange={(e) => setManuscriptName(e.target.value)}
          size="small"
        />
        <Button variant="contained" onClick={handleSave} disabled={!manuscriptName || !manuscript.length || savingManuscript} sx={{ mr: 1 }}>
          Save
        </Button>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Load</InputLabel>
          <Select
            label="Load"
            onChange={(e) => handleLoad(e.target.value as string)}
            value=""
            disabled={loadingSpecificManuscript || loadingManuscripts}
            sx={{ minWidth: 120 }}
          >
            {loadingSpecificManuscript || loadingManuscripts ? (
              <MenuItem disabled><CircularProgress size={20} /></MenuItem>
            ) : (
              savedManuscripts.map((name) => (
                <MenuItem key={name} value={name}>
                  {name}
                </MenuItem>
              ))
            )}
          </Select>
        </FormControl>
      </Box>
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Select Function</InputLabel>
        <Select
          value={selectedFunction}
          label="Select Function"
          onChange={(e) => setSelectedFunction(e.target.value as string)}
          disabled={loadingFunctions}
        >
          {loadingFunctions ? (
            <MenuItem disabled><CircularProgress size={20} /></MenuItem>
          ) : (
            Object.keys(functions).map((funcName) => (
              <MenuItem key={funcName} value={funcName}>
                {functions[funcName].name}
              </MenuItem>
            ))
          )}
        </Select>
      </FormControl>
      <Button variant="contained" onClick={handleAddStep} disabled={!selectedFunction}>
        Add Step
      </Button>
      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="manuscript-steps">
          {(provided) => (
            <List sx={{ mt: 2 }} {...provided.droppableProps} ref={provided.innerRef}>
              {manuscript.map((step, index) => (
                <Draggable key={step.name + index} draggableId={step.name + index} index={index}>
                  {(provided) => (
                    <ListItem
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                      sx={{ display: 'block', border: '1px solid #e0e0e0', mb: 1, p: 1 }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <ListItemText primary={`${index + 1}. ${step.name}`} />
                        {functions[step.name]?.doc && (
                          <Tooltip title={functions[step.name].doc} placement="right">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                      {Object.keys(functions[step.name]?.params || {}).map((paramName: string) => (
                        renderParamInput(step, index, paramName)
                      ))}
                    </ListItem>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </List>
          )}
        </Droppable>
      </DragDropContext>
    </Paper>
  );
};

export default ManuscriptEditor;