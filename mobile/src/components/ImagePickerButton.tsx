import React from "react";
import {
  Alert,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import Animated from "react-native-reanimated";
import { useScalePress } from "../animations";
import { colors, typography, spacing, radii } from "../theme";

interface Props {
  uri: string | null;
  onPick: (uri: string) => void;
  label?: string;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function ImagePickerButton({ uri, onPick, label }: Props) {
  const { animatedStyle, onPressIn, onPressOut } = useScalePress();

  const pick = () => {
    Alert.alert("Select Image", "Choose a source", [
      {
        text: "Camera",
        onPress: async () => {
          const perm = await ImagePicker.requestCameraPermissionsAsync();
          if (!perm.granted) return Alert.alert("Camera permission required");
          const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
          if (!result.canceled) onPick(result.assets[0].uri);
        },
      },
      {
        text: "Gallery",
        onPress: async () => {
          const perm =
            await ImagePicker.requestMediaLibraryPermissionsAsync();
          if (!perm.granted) return Alert.alert("Gallery permission required");
          const result = await ImagePicker.launchImageLibraryAsync({
            quality: 0.8,
          });
          if (!result.canceled) onPick(result.assets[0].uri);
        },
      },
      { text: "Cancel", style: "cancel" },
    ]);
  };

  return (
    <AnimatedPressable
      style={[styles.container, animatedStyle]}
      onPress={pick}
      onPressIn={onPressIn}
      onPressOut={onPressOut}
    >
      {uri ? (
        <View style={styles.imageWrap}>
          <Image source={{ uri }} style={styles.image} />
          <View style={styles.changeBadge}>
            <Ionicons name="camera-outline" size={14} color={colors.text} />
            <Text style={styles.changeText}>Change</Text>
          </View>
        </View>
      ) : (
        <View style={styles.placeholder}>
          <View style={styles.iconCircle}>
            <Ionicons name="image-outline" size={32} color={colors.primary} />
          </View>
          <Text style={styles.label}>{label || "Pick Image"}</Text>
          <Text style={styles.hint}>Tap to select</Text>
        </View>
      )}
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: radii.lg,
    overflow: "hidden",
    backgroundColor: colors.primaryMuted,
    borderWidth: 1,
    borderColor: colors.border,
  },
  imageWrap: {
    flex: 1,
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },
  changeBadge: {
    position: "absolute",
    bottom: spacing.sm,
    right: spacing.sm,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.65)",
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    gap: 4,
  },
  changeText: {
    ...typography.caption,
    color: colors.text,
    fontWeight: "600",
  },
  placeholder: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: spacing.sm,
  },
  iconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "rgba(124, 108, 255, 0.15)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.xs,
  },
  label: {
    ...typography.h3,
    color: colors.textSecondary,
  },
  hint: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
