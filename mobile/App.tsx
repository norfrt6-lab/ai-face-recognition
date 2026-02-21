import React from "react";
import { StatusBar } from "expo-status-bar";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";

import HomeScreen from "./src/screens/HomeScreen";
import SwapScreen from "./src/screens/SwapScreen";
import RecognizeScreen from "./src/screens/RecognizeScreen";
import FacesScreen from "./src/screens/FacesScreen";
import { colors, typography } from "./src/theme";

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <StatusBar style="light" />
        <Tab.Navigator
          screenOptions={({ route }) => ({
            headerShown: false,
            tabBarStyle: {
              backgroundColor: colors.background,
              borderTopColor: colors.border,
              borderTopWidth: 0.5,
              height: 64,
              paddingBottom: 8,
              paddingTop: 6,
            },
            tabBarActiveTintColor: colors.primary,
            tabBarInactiveTintColor: colors.textMuted,
            tabBarLabelStyle: {
              ...typography.caption,
              fontWeight: "600",
            },
            tabBarIcon: ({ color, size }) => {
              let iconName: keyof typeof Ionicons.glyphMap = "home";
              if (route.name === "Home") iconName = "home";
              else if (route.name === "Swap") iconName = "swap-horizontal";
              else if (route.name === "Recognize") iconName = "scan";
              else if (route.name === "Faces") iconName = "people";
              return <Ionicons name={iconName} size={size} color={color} />;
            },
          })}
        >
          <Tab.Screen name="Home" component={HomeScreen} />
          <Tab.Screen name="Swap" component={SwapScreen} />
          <Tab.Screen name="Recognize" component={RecognizeScreen} />
          <Tab.Screen name="Faces" component={FacesScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}
