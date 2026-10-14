import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  center: [20.5937, 78.9629],
  zoom: 5,
  selectedCoordinates: null,
  mapMode: 'explore', // 'explore' | 'selection' | 'price'
  selectedArea: null,
};

export const mapSlice = createSlice({
  name: 'map',
  initialState,
  reducers: {
    setCenter: (state, action) => {
      state.center = action.payload;
    },
    setZoom: (state, action) => {
      state.zoom = action.payload;
    },
    setSelectedCoordinates: (state, action) => {
      state.selectedCoordinates = action.payload;
    },
    setSelectedArea: (state, action) => {
      state.selectedArea = action.payload;
    },
    setMapMode: (state, action) => {
      state.mapMode = action.payload;
    },
    resetMap: (state) => {
      state.center = [20.5937, 78.9629];
      state.zoom = 5;
      state.selectedCoordinates = null;
      state.selectedArea = null;
      state.mapMode = 'explore';
    },
  },
});

export const {
  setCenter,
  setZoom,
  setSelectedCoordinates,
  setSelectedArea,
  setMapMode,
  resetMap,
} = mapSlice.actions;

export default mapSlice.reducer;
