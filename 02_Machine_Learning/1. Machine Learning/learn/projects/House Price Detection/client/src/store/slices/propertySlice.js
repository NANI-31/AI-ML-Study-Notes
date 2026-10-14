import { createSlice } from '@reduxjs/toolkit';
import propertyService from '../../services/propertyService';

const initialState = {
  formData: { ...propertyService.defaultPropertyForm },
  isDirty: false,
};

export const propertySlice = createSlice({
  name: 'property',
  initialState,
  reducers: {
    updateFormField: (state, action) => {
      const { field, value } = action.payload;
      state.formData[field] = value;
      state.isDirty = true;
    },
    updateMultipleFields: (state, action) => {
      state.formData = {
        ...state.formData,
        ...action.payload,
      };
      state.isDirty = true;
    },
    resetPropertyForm: (state) => {
      state.formData = { ...propertyService.defaultPropertyForm };
      state.isDirty = false;
    },
  },
});

export const { updateFormField, updateMultipleFields, resetPropertyForm } =
  propertySlice.actions;

export default propertySlice.reducer;
