import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2', // A standard blue, can be adjusted
    },
    secondary: {
      main: '#dc004e', // A standard red, can be adjusted
    },
    background: {
      default: '#f4f6f8', // Light grey background for a clean look
      paper: '#ffffff', // White background for cards/papers
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h6: {
      fontWeight: 600,
    },
    subtitle1: {
      fontWeight: 500,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff', // White app bar
          color: '#333333', // Dark text for app bar
          boxShadow: '0px 1px 5px rgba(0, 0, 0, 0.1)', // Subtle shadow
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '8px', // Slightly rounded corners for cards
          boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.05)', // Subtle shadow for depth
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '4px',
          textTransform: 'none', // Keep button text as is
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined', // Default to outlined text fields
        size: 'small',
      },
    },
    MuiSelect: {
      defaultProps: {
        variant: 'outlined', // Default to outlined selects
        size: 'small',
      },
    },
    MuiInputLabel: {
      defaultProps: {
        shrink: true, // Always shrink labels
      },
    },
  },
});

export default theme;
