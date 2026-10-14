import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import predictionService from '../../services/predictionService';

export const predictPrice = createAsyncThunk(
  'prediction/predictPrice',
  async (propertyPayload, { rejectWithValue }) => {
    try {
      const data = await predictionService.predictHousePrice(propertyPayload);
      return data;
    } catch (err) {
      return rejectWithValue(err.message || 'Failed to generate property prediction.');
    }
  }
);

const initialState = {
  result: null,
  submittedProperty: null,
  loading: false,
  error: null,
  lastUpdated: null,
};

export const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    clearPrediction: (state) => {
      state.result = null;
      state.submittedProperty = null;
      state.loading = false;
      state.error = null;
      state.lastUpdated = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(predictPrice.pending, (state, action) => {
        state.loading = true;
        state.error = null;
        state.submittedProperty = action.meta.arg;
      })
      .addCase(predictPrice.fulfilled, (state, action) => {
        state.loading = false;
        state.result = action.payload;
        state.error = null;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(predictPrice.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || 'An error occurred during property prediction.';
      });
  },
});

export const { clearPrediction } = predictionSlice.actions;

export default predictionSlice.reducer;
