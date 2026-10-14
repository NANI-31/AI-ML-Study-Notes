import { configureStore } from '@reduxjs/toolkit';
import locationReducer from './slices/locationSlice';
import mapReducer from './slices/mapSlice';
import propertyReducer from './slices/propertySlice';
import predictionReducer from './slices/predictionSlice';

export const store = configureStore({
  reducer: {
    location: locationReducer,
    map: mapReducer,
    property: propertyReducer,
    prediction: predictionReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

export default store;
