#include <gtest/gtest.h>
#include "test_message.pb.h"

// Test to verify that we can construct a protobuf message
TEST(ProtobufTest, CanConstructProto) {
    amswarm::TestMessage message;
    
    // Set some fields
    message.set_name("test_name");
    message.set_id(42);
    message.set_value(3.14159);
    
    // Verify the fields were set correctly
    EXPECT_EQ(message.name(), "test_name");
    EXPECT_EQ(message.id(), 42);
    EXPECT_DOUBLE_EQ(message.value(), 3.14159);
}

// Test serialization and deserialization
TEST(ProtobufTest, SerializeAndDeserialize) {
    amswarm::TestMessage original;
    original.set_name("serialization_test");
    original.set_id(123);
    original.set_value(2.71828);
    
    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));
    EXPECT_FALSE(serialized.empty());
    
    // Deserialize from string
    amswarm::TestMessage deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));
    
    // Verify the deserialized message matches the original
    EXPECT_EQ(deserialized.name(), original.name());
    EXPECT_EQ(deserialized.id(), original.id());
    EXPECT_DOUBLE_EQ(deserialized.value(), original.value());
}
