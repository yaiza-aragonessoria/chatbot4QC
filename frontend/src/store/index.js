import {configureStore} from "@reduxjs/toolkit";
import authSlice from "../features/slices/authSlice";
import {userProfileSlice} from "../features/slices/userSlice";

const store = configureStore({
    reducer: {
        auth: authSlice.reducer,
        userProfile: userProfileSlice.reducer
    }
})

export default store