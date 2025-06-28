import { useState, useEffect } from 'react';

// Define the User type based on the Django serializer
interface User {
  id: number;
  email: string;
  name: string | null;
  // Add other fields from the serializer if needed
}

// Define the hook's return type
interface UseUserReturn {
  user: User | null;
  isLoading: boolean;
  error: Error | null;
}

const useUser = (): UseUserReturn => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchUser = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('/accounts/api/me/');
        if (!response.ok) {
          // If the user is not logged in (e.g., 401 Unauthorized), handle it gracefully
          if (response.status === 401 || response.status === 403) {
            console.log('User not authenticated.');
            setUser(null); // Ensure user is null if not authenticated
            return; 
          }
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: User = await response.json();
        setUser(data);
      } catch (err) {
        setError(err as Error);
        console.error("Failed to fetch user data:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUser();
  }, []); // Empty dependency array means this runs once on mount

  return { user, isLoading, error };
};

export default useUser;
